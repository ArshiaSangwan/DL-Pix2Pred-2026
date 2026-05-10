#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


SCORE_COLUMNS = ["score_A", "score_B", "score_C", "score_D", "score_E"]


def parse_weights(values, n):
    if not values:
        return [1.0] * n
    if len(values) != n:
        raise ValueError(f"--weights must have {n} values, got {len(values)}")
    return [float(value) for value in values]


def validate_submission(submission_df, target_df):
    if list(submission_df.columns) != ["id", "answer"]:
        raise ValueError("submission must have exactly columns: id, answer")
    if len(submission_df) != len(target_df):
        raise ValueError(f"row count mismatch: {len(submission_df)} != {len(target_df)}")
    if set(submission_df["id"]) != set(target_df["id"]):
        raise ValueError("submission ids do not match target ids")
    merged = target_df[["id", "num_choices"]].merge(submission_df, on="id", how="left")
    if merged["answer"].isna().any():
        raise ValueError("missing answer in submission")
    merged["answer"] = merged["answer"].astype(int)
    invalid = (merged["answer"] < 0) | (merged["answer"] >= merged["num_choices"])
    if invalid.any():
        bad_id = merged.loc[invalid, "id"].iloc[0]
        raise ValueError(f"answer out of range for {bad_id}")


def load_score_file(path, weight):
    df = pd.read_csv(path)
    missing = [column for column in ["id", "num_choices", *SCORE_COLUMNS] if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    keep = df[["id", "num_choices", *SCORE_COLUMNS]].copy()
    for column in SCORE_COLUMNS:
        keep[column] = keep[column].fillna(-1e9) * weight
    return keep


def main():
    parser = argparse.ArgumentParser(description="Average per-choice score dumps into a validated submission.")
    parser.add_argument("--test_csv", required=True, help="Target CSV with id and num_choices")
    parser.add_argument("--score_csvs", nargs="+", required=True, help="Score dumps from src/inference.py")
    parser.add_argument("--weights", nargs="*", default=None, help="Optional weight per score CSV")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--score_out", default=None)
    args = parser.parse_args()

    target_df = pd.read_csv(args.test_csv)
    weights = parse_weights(args.weights, len(args.score_csvs))
    merged = None
    total_weight = sum(weights)
    for idx, (score_csv, weight) in enumerate(zip(args.score_csvs, weights)):
        scores = load_score_file(score_csv, weight)
        renamed = scores.rename(columns={column: f"{column}_{idx}" for column in SCORE_COLUMNS})
        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed.drop(columns=["num_choices"]), on="id", how="inner")

    if merged is None or len(merged) != len(target_df):
        raise ValueError("score CSVs did not cover every target row")

    out_scores = target_df[["id", "num_choices"]].merge(merged.drop(columns=["num_choices"]), on="id", how="left")
    for column in SCORE_COLUMNS:
        per_file_cols = [f"{column}_{idx}" for idx in range(len(args.score_csvs))]
        out_scores[column] = out_scores[per_file_cols].sum(axis=1) / total_weight

    predictions = []
    for _, row in out_scores.iterrows():
        num_choices = int(row["num_choices"])
        scores = [float(row[column]) for column in SCORE_COLUMNS[:num_choices]]
        predictions.append(int(max(range(num_choices), key=lambda idx: scores[idx])))

    submission_df = pd.DataFrame({"id": out_scores["id"], "answer": predictions})
    validate_submission(submission_df, target_df)
    submission_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

    if args.score_out:
        compact_scores = out_scores[["id", "num_choices", *SCORE_COLUMNS]].copy()
        compact_scores["answer"] = predictions
        if "answer" in target_df.columns:
            labels = target_df[["id", "answer"]].rename(columns={"answer": "truth"})
            compact_scores = compact_scores.merge(labels, on="id", how="left")
            compact_scores["correct"] = compact_scores["answer"].astype(int) == compact_scores["truth"].astype(int)
            print(f"Accuracy: {compact_scores['correct'].mean():.4f}")
        Path(args.score_out).parent.mkdir(parents=True, exist_ok=True)
        compact_scores.to_csv(args.score_out, index=False)
        print(f"Wrote {args.score_out}")


if __name__ == "__main__":
    main()
