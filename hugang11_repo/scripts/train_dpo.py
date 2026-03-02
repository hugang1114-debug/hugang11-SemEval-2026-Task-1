#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc

import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

from humor_pipeline.prompts import format_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DPO model.")
    parser.add_argument("--model-path", required=True, help="Path to the SFT adapter checkpoint.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gc.collect()
    torch.cuda.empty_cache()

    PatchDPOTrainer()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")

    def to_dpo_format(example):
        return {
            "prompt": format_prompt(example["instruction"], example["input"], ""),
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    dataset = dataset.map(to_dpo_format)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_seq_length,
        gradient_checkpointing=True,
        beta=args.beta,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="steps",
        save_steps=50,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Compatibility patch for some TRL / Unsloth versions.
    original_log = trainer.log

    def patched_log(logs, start_time=None):
        return original_log(logs)

    trainer.log = patched_log
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
