#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


CKPT_RE = re.compile(r"ckpt_ep(?P<epoch>\d+)_acc(?P<acc>\d+(?:\.\d+)?)$")


def parse_checkpoint(path: Path):
    match = CKPT_RE.search(path.name)
    if not match:
        return None
    return {
        "path": path,
        "epoch": int(match.group("epoch")),
        "score": float(match.group("acc")),
    }


def main():
    parser = argparse.ArgumentParser(description="Select top final-fit adapter checkpoints by saved checkpoint score.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_epoch", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--include_best", action="store_true")
    parser.add_argument("--format", choices=["lines", "space"], default="lines")
    args = parser.parse_args()

    checkpoint_dir = Path(args.run_dir) / "checkpoints"
    items = []
    if checkpoint_dir.exists():
        for path in checkpoint_dir.iterdir():
            if not path.is_dir():
                continue
            item = parse_checkpoint(path)
            if item is None:
                continue
            if item["epoch"] < args.min_epoch:
                continue
            if args.max_epoch is not None and item["epoch"] > args.max_epoch:
                continue
            items.append(item)
    items.sort(key=lambda item: (item["score"], item["epoch"]), reverse=True)

    selected = [str(item["path"]) for item in items[: max(1, int(args.top_k))]]
    best_model = Path(args.run_dir) / "best_model"
    if args.include_best and best_model.exists():
        selected.insert(0, str(best_model))

    deduped = []
    seen = set()
    for path in selected:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)

    if args.format == "space":
        print(" ".join(deduped))
    else:
        for path in deduped:
            print(path)


if __name__ == "__main__":
    main()
