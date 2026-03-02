#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from humor_pipeline.prompts import build_instruction_and_input


NEWS_SYSTEM_PROMPT = """你是一个幽默专栏作家。请针对新闻标题，按以下格式输出：
1. 【思考逻辑】：分析新闻荒谬点，确定幽默技巧（反讽/夸张等）。
2. 【神吐槽】：写一个简短有力的爆笑段子。"""

WORDS_SYSTEM_PROMPT = """你是一个脑洞大开的脱口秀演员。
任务：请用给定的两个词造一个荒谬好笑的段子。
要求：
1. 段子中必须包含这两个词。
2. 按以下格式输出：
   【思考逻辑】：分析如何把这两个风马牛不相及的词联系起来，构建场景。
   【神吐槽】：生成的段子（必须包含给定词）。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CoT-SFT JSONL data with a teacher model.")
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Please set --api-key or OPENAI_API_KEY / DEEPSEEK_API_KEY.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    df = pd.read_csv(args.input_tsv, sep="\t", dtype=str, keep_default_na=False)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Teacher rewriting"):
        row_dict = row.to_dict()
        try:
            instruction, user_input = build_instruction_and_input(row_dict)
        except ValueError:
            continue

        if row_dict.get("headline", "") and row_dict.get("headline", "") != "-":
            system_prompt = NEWS_SYSTEM_PROMPT
            user_prompt = user_input
        else:
            word1 = str(row_dict.get("word1", "") or "").strip()
            word2 = str(row_dict.get("word2", "") or "").strip()
            system_prompt = WORDS_SYSTEM_PROMPT.replace("给定的两个词", f"给定的两个词（{word1}、{word2}）").replace("给定词", f"{word1} 和 {word2}")
            user_prompt = user_input

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            output_text = response.choices[0].message.content
        except Exception as exc:
            print(f"[WARN] Teacher request failed: {exc}")
            continue

        rows.append({
            "instruction": instruction,
            "input": user_input,
            "output": output_text,
        })

    with output_path.open("w", encoding="utf-8") as f:
        for item in rows:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
