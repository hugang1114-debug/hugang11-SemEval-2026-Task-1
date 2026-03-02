#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel

from humor_pipeline.cleaning import clean_generated_text, save_submission
from humor_pipeline.prompts import build_instruction_and_input, format_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a competition submission TSV.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--strict-cot", action="store_true", help="Append the strict structured decoding constraint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_tsv, sep="\t", dtype=str, keep_default_na=False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        row_dict = row.to_dict()
        try:
            instruction, user_input = build_instruction_and_input(row_dict, strict=args.strict_cot)
        except ValueError:
            outputs.append("Error")
            continue

        inputs = tokenizer([format_prompt(instruction, user_input, "")], return_tensors="pt").to("cuda")
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_cache=True,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        if "### Response:\n" in decoded:
            decoded = decoded.split("### Response:\n", 1)[1]
        outputs.append(clean_generated_text(decoded))

    result = df.copy()
    result["text"] = outputs
    save_submission(result[["id", "text"]], args.output_tsv)
    print(f"Saved submission to {args.output_tsv}")


if __name__ == "__main__":
    main()
