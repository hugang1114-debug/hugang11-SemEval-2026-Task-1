# hugang11 at SemEval-2026 Task 1 (MWAHAHA), Subtask A (Chinese)

This repository contains the code for the **hugang11** system submitted to **SemEval-2026 Task 1: MWAHAHA -- Competition on Humor Generation**, **Subtask A (Chinese)**.

Our system is a practical multi-stage pipeline for Chinese humor generation built around:

- **CoT-SFT** (chain-of-thought-augmented supervised fine-tuning)
- **teacher-constructed DPO** (Direct Preference Optimization)
- **structured decoding**
- **deterministic post-processing** for competition-ready submissions

The implementation follows the system described in the accompanying SemEval system paper.

---

## Overview

The pipeline consists of the following stages:

1. Build **CoT-SFT supervision data** from the official training set using a stronger teacher model.
2. Build **preference pairs** for DPO by pairing teacher outputs with weaker baseline outputs.
3. Train a **CoT-SFT model** on teacher-rewritten data.
4. Train a **DPO model** on the preference pairs.
5. Generate a raw TSV submission with **structured decoding**.
6. Clean the raw generations with a **deterministic post-processing pipeline**.

The final system is based on **Qwen2.5-7B-Instruct**, **LoRA**, **4-bit loading**, and **Unsloth**.

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── run.sh
├── .gitignore
├── scripts/
│   ├── prepare_cot_sft_data.py
│   ├── build_dpo_pairs.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── generate_submission.py
│   └── clean_submission.py
├── src/
│   └── humor_pipeline/
│       ├── __init__.py
│       ├── prompts.py
│       └── cleaning.py
└── data/
    └── samples/
        ├── sample_train.tsv
        ├── sample_test.tsv
        ├── sample_cot_sft.jsonl
        ├── sample_cot_dpo.jsonl
        ├── sample_plain_dpo.jsonl
        ├── sample_raw_submission.tsv
        └── sample_clean_submission.tsv
```

---

## Environment

Recommended environment:

- Python 3.10+
- CUDA-enabled GPU
- A100 or similar high-memory GPU recommended for training and inference

Install dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH=./src
```

---

## Quick Start

### Option 1: Run the main stages with `run.sh`

```bash
bash run.sh train-sft
bash run.sh train-dpo
bash run.sh generate
bash run.sh clean
```

If you already have the processed JSONL files prepared, the following command runs the main training + inference pipeline end to end:

```bash
bash run.sh all
```

### Option 2: Run the scripts manually

```bash
python scripts/train_sft.py \
  --train-jsonl data/processed/train_cot_sft_v2.jsonl \
  --output-dir checkpoints/sft_cot_final

python scripts/train_dpo.py \
  --model-path checkpoints/sft_cot_final \
  --train-jsonl data/processed/train_cot_dpo_v2.jsonl \
  --output-dir checkpoints/dpo_cot_final

python scripts/generate_submission.py \
  --model-path checkpoints/dpo_cot_final \
  --input-tsv data/raw/300_final_test_sets.tsv \
  --output-tsv data/submissions/raw_submission.tsv \
  --strict-cot

python scripts/clean_submission.py \
  --input-tsv data/submissions/raw_submission.tsv \
  --output-tsv data/submissions/cleaned_submission.tsv
```

---

## Data Layout

This repository does **not** depend on Google Drive mounting or Colab-specific path logic. All scripts use explicit local file paths.

Suggested local layout for full experiments:

```text
data/
├── raw/
│   ├── 1000_training_sets.tsv
│   ├── 300_final_test_sets.tsv
│   └── baseline_train_output.tsv
├── processed/
│   ├── train_cot_sft_v2.jsonl
│   └── train_cot_dpo_v2.jsonl
└── submissions/
    ├── raw_submission.tsv
    └── cleaned_submission.tsv
```

The repository includes only **small format examples** under `data/samples/` so that users can inspect the expected file structure without redistributing the full task data.

---

## Sample Files Included

The following minimal example files are included under `data/samples/`:

- `sample_train.tsv`: small example of official-style training TSV
- `sample_test.tsv`: small example of official-style test TSV
- `sample_cot_sft.jsonl`: small example of CoT-SFT supervision data
- `sample_cot_dpo.jsonl`: small example of CoT-DPO preference pairs
- `sample_plain_dpo.jsonl`: small example of plain DPO pairs
- `sample_raw_submission.tsv`: noisy raw generation example
- `sample_clean_submission.tsv`: cleaned submission example

These files are for **format illustration only**, not for benchmarking.

---

## Expected Input Formats

### Official TSV data

Each row should contain an `id` field and either:

- a `headline` column for news-title prompts, or
- `word1` and `word2` columns for word-pair prompts.

### CoT-SFT JSONL

Each line should look like:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

### CoT-DPO JSONL

Each line should look like:

```json
{"instruction": "...", "input": "...", "chosen": "...", "rejected": "..."}
```

### Plain DPO JSONL

Each line may also be stored in a prompt-based format:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

---

## Main Scripts

### `prepare_cot_sft_data.py`
Uses a teacher model API to rewrite official task inputs into CoT-style supervision data.

### `build_dpo_pairs.py`
Builds DPO pairs by matching teacher outputs with weaker baseline outputs for the same prompt.

### `train_sft.py`
Runs LoRA-based CoT-SFT training with the Unsloth / Hugging Face stack.

### `train_dpo.py`
Runs DPO training on the preference pairs starting from the SFT adapter.

### `generate_submission.py`
Loads the trained model and generates a TSV submission with structured decoding.

### `clean_submission.py`
Cleans malformed outputs and exports a competition-ready TSV.

---

## Reproducing the Paper Setup

Key settings used in the SemEval system paper:

- Base model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- LoRA: rank 16, alpha 16, dropout 0.05
- Chinese training data: 1,000 instances
- CoT-SFT data: 1,000 examples
- DPO data: 1,000 preference pairs
- DPO beta: 0.1
- Decoding: temperature 0.7, top-p 0.9, repetition penalty 1.2
- Final large-scale inference: NVIDIA A100 40GB

---

## Post-processing Behavior

The cleaning pipeline includes:

- extraction of the final roast segment
- removal of leaked reasoning traces
- truncation of repeated gibberish loops
- removal of markdown residue and malformed quotes
- robust TSV parsing and re-saving

This stage is important because raw generations may contain:

- leaked reasoning blocks
- repetition artifacts
- broken TSV rows
- malformed quotes or prompt residue

---

## Security Notes

If you use API-based teacher generation, **do not commit API keys to GitHub**.
Always pass them through environment variables, for example:

```bash
export DEEPSEEK_API_KEY=your_key_here
```

Model checkpoints, private credentials, and restricted competition data should not be committed directly to the repository.

---

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Citation

If you use this code or build on this repository, please cite the SemEval task overview paper and the corresponding system paper.

```bibtex
@inproceedings{semeval2026mwahaha,
  title={{SemEval-2026 Task 1: MWAHAHA, Models Write Automatic Humor And Humans Annotate}},
  author={Castro, Santiago and Chiruzzo, Luis and G{\'o}ngora, Santiago and Rahili, Salar and Deng, Naihao and Sastre, Ignacio and Amoroso, Victoria and Rey, Guillermo and Ros{\'a}, Aiala and Moncecchi, Guillermo and Meaney, J. A. and Prada, Juan Jos{\'e} and Mihalcea, Rada},
  booktitle={Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year={2026}
}
```

---

## Contact

**Gang Hu**  
Yunnan University  
Kunming, China  
Email: `12025215204@stu.ynu.edu.cn`
