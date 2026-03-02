#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=./src:${PYTHONPATH:-}

# Usage:
#   bash run.sh prepare-cot-sft
#   bash run.sh build-dpo-pairs
#   bash run.sh train-sft
#   bash run.sh train-dpo
#   bash run.sh generate
#   bash run.sh clean
#   bash run.sh all
#
# Before running:
#   1. Put your full task files under data/raw/
#   2. Export DEEPSEEK_API_KEY (only needed for prepare-cot-sft)
#   3. Adjust file paths below if needed

COMMAND=${1:-help}

RAW_TRAIN=${RAW_TRAIN:-data/raw/1000_training_sets.tsv}
RAW_TEST=${RAW_TEST:-data/raw/300_final_test_sets.tsv}
BASELINE_TRAIN_OUTPUT=${BASELINE_TRAIN_OUTPUT:-data/raw/baseline_train_output.tsv}
COT_SFT_JSONL=${COT_SFT_JSONL:-data/processed/train_cot_sft_v2.jsonl}
COT_DPO_JSONL=${COT_DPO_JSONL:-data/processed/train_cot_dpo_v2.jsonl}
SFT_OUT=${SFT_OUT:-checkpoints/sft_cot_final}
DPO_OUT=${DPO_OUT:-checkpoints/dpo_cot_final}
RAW_SUB=${RAW_SUB:-data/submissions/raw_submission.tsv}
CLEAN_SUB=${CLEAN_SUB:-data/submissions/cleaned_submission.tsv}

mkdir -p data/processed data/submissions checkpoints

case "$COMMAND" in
  prepare-cot-sft)
    python scripts/prepare_cot_sft_data.py \
      --input-tsv "$RAW_TRAIN" \
      --output-jsonl "$COT_SFT_JSONL"
    ;;

  build-dpo-pairs)
    python scripts/build_dpo_pairs.py \
      --teacher-jsonl "$COT_SFT_JSONL" \
      --baseline-tsv "$BASELINE_TRAIN_OUTPUT" \
      --output-jsonl "$COT_DPO_JSONL"
    ;;

  train-sft)
    python scripts/train_sft.py \
      --train-jsonl "$COT_SFT_JSONL" \
      --output-dir "$SFT_OUT"
    ;;

  train-dpo)
    python scripts/train_dpo.py \
      --model-path "$SFT_OUT" \
      --train-jsonl "$COT_DPO_JSONL" \
      --output-dir "$DPO_OUT"
    ;;

  generate)
    python scripts/generate_submission.py \
      --model-path "$DPO_OUT" \
      --input-tsv "$RAW_TEST" \
      --output-tsv "$RAW_SUB" \
      --strict-cot
    ;;

  clean)
    python scripts/clean_submission.py \
      --input-tsv "$RAW_SUB" \
      --output-tsv "$CLEAN_SUB"
    ;;

  all)
    python scripts/train_sft.py \
      --train-jsonl "$COT_SFT_JSONL" \
      --output-dir "$SFT_OUT"
    python scripts/train_dpo.py \
      --model-path "$SFT_OUT" \
      --train-jsonl "$COT_DPO_JSONL" \
      --output-dir "$DPO_OUT"
    python scripts/generate_submission.py \
      --model-path "$DPO_OUT" \
      --input-tsv "$RAW_TEST" \
      --output-tsv "$RAW_SUB" \
      --strict-cot
    python scripts/clean_submission.py \
      --input-tsv "$RAW_SUB" \
      --output-tsv "$CLEAN_SUB"
    ;;

  help|--help|-h)
    sed -n '1,120p' "$0"
    ;;

  *)
    echo "Unknown command: $COMMAND"
    exit 1
    ;;
esac
