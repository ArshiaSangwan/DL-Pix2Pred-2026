#!/usr/bin/env python3
import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCORE_COLUMNS = ["score_A", "score_B", "score_C", "score_D", "score_E"]


def find_score_csvs(score_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(score_dir.rglob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if {"id", "num_choices", *SCORE_COLUMNS}.issubset(set(header.columns)):
            paths.append(path)
    return paths


def load_score_matrix(path: Path, labels: pd.DataFrame) -> np.ndarray:
    scores = pd.read_csv(path)
    missing = [column for column in ["id", "num_choices", *SCORE_COLUMNS] if column not in scores.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    merged = labels[["id", "num_choices"]].merge(scores[["id", *SCORE_COLUMNS]], on="id", how="left")
    if merged[SCORE_COLUMNS].isna().all(axis=1).any():
        bad_id = merged.loc[merged[SCORE_COLUMNS].isna().all(axis=1), "id"].iloc[0]
        raise ValueError(f"{path} does not contain scores for id={bad_id}")
    matrix = merged[SCORE_COLUMNS].fillna(-1e9).to_numpy(dtype=np.float64)
    return matrix


def predict(matrix: np.ndarray, num_choices: np.ndarray) -> np.ndarray:
    masked = matrix.copy()
    for row_idx, n_choices in enumerate(num_choices):
        masked[row_idx, int(n_choices) :] = -1e12
    return masked.argmax(axis=1).astype(int)


def accuracy(matrix: np.ndarray, labels: np.ndarray, num_choices: np.ndarray) -> float:
    return float((predict(matrix, num_choices) == labels).mean())


def simplex_weights(size: int, steps: int):
    if size == 1:
        yield (1.0,)
        return
    for raw in itertools.product(range(steps + 1), repeat=size):
        total = sum(raw)
        if total <= 0:
            continue
        yield tuple(value / total for value in raw)


def make_candidate_sets(order: list[int], max_size: int, pool_size: int):
    pool = order[: max(1, min(pool_size, len(order)))]
    for size in range(1, min(max_size, len(pool)) + 1):
        for combo in itertools.combinations(pool, size):
            yield combo


def main():
    parser = argparse.ArgumentParser(description="Search validation-gated score ensembles.")
    parser.add_argument("--labels", required=True, help="Validation CSV with id, answer, and num_choices.")
    parser.add_argument("--score_dir", default=None, help="Directory containing score CSVs from src/inference.py.")
    parser.add_argument("--score_csvs", nargs="*", default=None, help="Optional explicit score CSV list.")
    parser.add_argument("--out", default="runs/eval/best_ensemble.json")
    parser.add_argument("--pred_out", default=None, help="Optional CSV of predictions from the best ensemble.")
    parser.add_argument("--weight_steps", type=int, default=4, help="Grid granularity. 4 gives 0.25 increments.")
    parser.add_argument("--max_ensemble_size", type=int, default=4)
    parser.add_argument("--candidate_pool", type=int, default=8)
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels)
    required = {"id", "answer", "num_choices"}
    missing = required.difference(labels_df.columns)
    if missing:
        raise ValueError(f"{args.labels} is missing columns: {sorted(missing)}")

    if args.score_csvs:
        score_paths = [Path(path) for path in args.score_csvs]
    elif args.score_dir:
        score_paths = find_score_csvs(Path(args.score_dir))
    else:
        raise ValueError("Pass --score_dir or --score_csvs.")
    if not score_paths:
        raise ValueError("No score CSVs found.")

    y_true = labels_df["answer"].astype(int).to_numpy()
    num_choices = labels_df["num_choices"].astype(int).to_numpy()
    valid_paths = []
    matrices = []
    skipped = []
    for path in score_paths:
        try:
            matrix = load_score_matrix(path, labels_df)
        except ValueError as exc:
            skipped.append({"score_csv": str(path), "reason": str(exc)})
            print(f"Skipping incompatible score CSV: {path} ({exc})")
            continue
        valid_paths.append(path)
        matrices.append(matrix)
    score_paths = valid_paths
    if not matrices:
        raise ValueError("No score CSVs matched the validation labels.")

    individual = []
    for idx, (path, matrix) in enumerate(zip(score_paths, matrices)):
        acc = accuracy(matrix, y_true, num_choices)
        individual.append({"index": idx, "score_csv": str(path), "accuracy": acc})
    individual.sort(key=lambda item: item["accuracy"], reverse=True)
    order = [item["index"] for item in individual]

    best = {
        "accuracy": -1.0,
        "indices": [],
        "score_csvs": [],
        "weights": [],
        "rows": int(len(labels_df)),
    }

    evaluated = 0
    seen = set()
    for combo in make_candidate_sets(order, args.max_ensemble_size, args.candidate_pool):
        for weights in simplex_weights(len(combo), max(1, int(args.weight_steps))):
            key = (combo, tuple(round(weight, 6) for weight in weights))
            if key in seen:
                continue
            seen.add(key)
            weighted = np.zeros_like(matrices[0], dtype=np.float64)
            for matrix_idx, weight in zip(combo, weights):
                weighted += matrices[matrix_idx] * weight
            acc = accuracy(weighted, y_true, num_choices)
            evaluated += 1
            if acc > best["accuracy"]:
                best = {
                    "accuracy": acc,
                    "indices": list(combo),
                    "score_csvs": [str(score_paths[idx]) for idx in combo],
                    "weights": [float(weight) for weight in weights],
                    "rows": int(len(labels_df)),
                }

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": str(Path(args.labels)),
        "evaluated_recipes": evaluated,
        "best": best,
        "individual": individual,
        "skipped": skipped,
        "search": {
            "weight_steps": int(args.weight_steps),
            "max_ensemble_size": int(args.max_ensemble_size),
            "candidate_pool": int(args.candidate_pool),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best['accuracy']:.4f}")
    print(f"Best score CSVs: {best['score_csvs']}")
    print(f"Best weights: {best['weights']}")
    print(f"Wrote {out_path}")

    if args.pred_out:
        weighted = np.zeros_like(matrices[0], dtype=np.float64)
        for path, weight in zip(best["score_csvs"], best["weights"]):
            matrix_idx = score_paths.index(Path(path))
            weighted += matrices[matrix_idx] * weight
        pred_df = labels_df[["id", "num_choices", "answer"]].copy()
        pred_df["pred"] = predict(weighted, num_choices)
        pred_df["correct"] = pred_df["pred"].astype(int) == pred_df["answer"].astype(int)
        for idx, column in enumerate(SCORE_COLUMNS):
            pred_df[column] = weighted[:, idx]
        pred_path = Path(args.pred_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
