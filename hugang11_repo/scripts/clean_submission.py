#!/usr/bin/env python3
from __future__ import annotations

import argparse

from humor_pipeline.cleaning import clean_generated_text, robust_read_tsv, save_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a noisy TSV submission file.")
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-tsv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = robust_read_tsv(args.input_tsv)
    if "text" not in df.columns:
        raise ValueError("Input TSV must contain a 'text' column after parsing.")
    df["text"] = df["text"].apply(clean_generated_text)
    save_submission(df[["id", "text"]], args.output_tsv)
    print(f"Saved cleaned submission to {args.output_tsv}")


if __name__ == "__main__":
    main()
