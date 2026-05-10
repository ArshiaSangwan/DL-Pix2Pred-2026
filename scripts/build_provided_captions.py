#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CAPTION_RE = re.compile(
    r"(?:Figure|Image|Diagram|Picture|Photo|Map|Graph|Chart|Table)\s*:\s*([^\n]+)",
    re.I,
)


def clean_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    text = re.sub(r"\s*(?:For example|The image|This image).*$", "", text, flags=re.I).strip()
    if text and not text.endswith("."):
        text += "."
    return text


def extract_caption(row: pd.Series) -> tuple[str, str]:
    text = "\n".join(
        "" if pd.isna(row.get(field)) else str(row.get(field))
        for field in ["question", "hint", "lecture"]
    )
    match = CAPTION_RE.search(text)
    if not match:
        return "", "blank_no_provided_caption"
    caption = clean_caption(match.group(1))
    return caption, "provided" if caption else "blank_no_provided_caption"


def main():
    parser = argparse.ArgumentParser(description="Extract provided figure captions from competition rows.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--caption_col", default="caption")
    parser.add_argument("--source_col", default="caption_source")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    extracted = [extract_caption(row) for _, row in df.iterrows()]
    df[args.caption_col] = [caption for caption, _ in extracted]
    df[args.source_col] = [source for _, source in extracted]
    df[f"{args.caption_col}_generated"] = True

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    manifest = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "out_csv": str(out_path.resolve()),
        "caption_col": args.caption_col,
        "source_col": args.source_col,
        "rows": int(len(df)),
        "provided_caption_rows": int((df[args.source_col] == "provided").sum()),
        "blank_rows": int(df[args.caption_col].fillna("").eq("").sum()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
