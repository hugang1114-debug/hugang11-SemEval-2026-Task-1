#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from humor_pipeline.prompts import build_instruction_and_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs from teacher data and baseline generations.")
    parser.add_argument("--teacher-jsonl", required=True, help="Teacher-generated CoT SFT JSONL.")
    parser.add_argument("--baseline-tsv", required=True, help="Baseline generations TSV (submission.tsv or similar).")
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def load_teacher_map(path: str) -> dict[str, str]:
    teacher_map: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            key = item["instruction"] + item["input"]
            teacher_map[key] = item["output"]
    return teacher_map


def main() -> None:
    args = parse_args()
    teacher_map = load_teacher_map(args.teacher_jsonl)
    df = pd.read_csv(args.baseline_tsv, sep="\t", dtype=str, keep_default_na=False)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building DPO pairs"):
        row_dict = row.to_dict()
        try:
            instruction, user_input = build_instruction_and_input(row_dict)
        except ValueError:
            continue

        key = instruction + user_input
        chosen = teacher_map.get(key)
        rejected = str(row.get("generated_text", "") or row.get("text", "") or "").strip()
        if not chosen or not rejected or chosen == rejected:
            continue

        pairs.append({
            "instruction": instruction,
            "input": user_input,
            "chosen": chosen,
            "rejected": rejected,
        })

    with output_path.open("w", encoding="utf-8") as f:
        for item in pairs:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(pairs)} preference pairs to {output_path}")


if __name__ == "__main__":
    main()
