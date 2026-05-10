#!/usr/bin/env python3
import argparse

import pandas as pd


def print_group(df, field, min_count):
    if field not in df.columns:
        return
    grouped = df.groupby(field)["correct"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= min_count].sort_values("mean")
    if not grouped.empty:
        print(f"\nAccuracy by {field}:")
        print(grouped.to_string())


def main():
    parser = argparse.ArgumentParser(description="Summarize per-choice score dumps against labeled CSVs.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--min_count", type=int, default=10)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    labels = pd.read_csv(args.labels)
    df = scores.merge(labels, on="id", how="left", suffixes=("", "_label"))
    if "truth" not in df.columns:
        if "answer_label" in df.columns:
            df["truth"] = df["answer_label"]
        elif "answer" in labels.columns:
            df["truth"] = df["answer"]
        else:
            raise ValueError("No truth labels found in score dump or labels CSV")
    df["correct"] = df["answer"].astype(int) == df["truth"].astype(int)

    print(f"Rows: {len(df)}")
    print(f"Accuracy: {df['correct'].mean():.4f}")
    print_group(df, "num_choices", 1)
    print_group(df, "category", args.min_count)
    print_group(df, "topic", args.min_count)
    print_group(df, "subject", args.min_count)

    if "margin" in df.columns:
        bins = pd.cut(df["margin"], bins=[-float("inf"), 0.5, 1.0, 2.0, 4.0, float("inf")])
        margin_group = df.groupby(bins, observed=False)["correct"].agg(["mean", "count"])
        print("\nAccuracy by margin:")
        print(margin_group.to_string())


if __name__ == "__main__":
    main()
