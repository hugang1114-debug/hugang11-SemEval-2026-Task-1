from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

SPECIAL_TOKENS = ["<|im_end|>", "<|endoftext|>"]
GARBAGE_WORDS = ["jkwenk", "pirence", "prarzor", "ksam", "zychephyr", "ecerec"]


def clean_generated_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")

    # Truncate obvious repeated gibberish loops.
    match = re.search(r"((\w{3,})\2{4,})", text)
    if match:
        text = text[: match.start()]

    for garbage in GARBAGE_WORDS:
        if garbage in text:
            text = text.split(garbage)[0]

    # Prefer the final roast tag if present.
    roast_matches = list(re.finditer(r"【神吐槽】[:：]?\s*", text))
    if roast_matches:
        text = text[roast_matches[-1].end() :]
    else:
        # Remove reasoning traces when the roast tag is missing.
        text = re.sub(r"【思考逻辑】.*?(?=【|$)", "", text, flags=re.DOTALL)
        parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        if parts:
            text = parts[-1]

    text = text.replace("```markdown", "").replace("```", "")
    text = re.sub(r"^---.*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[:：\-\s]+", "", text)
    text = text.strip().strip('"').strip("'").strip()
    return text


def _parse_raw_tsv(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return pd.DataFrame(columns=["id", "text"])
    if lines[0].startswith("id\t"):
        lines = lines[1:]

    rows: List[Tuple[str, str]] = []
    current_id = None
    current_text: List[str] = []
    id_pattern = re.compile(r"^(zh_\d{4})\t?(.*)$")

    for line in lines:
        match = id_pattern.match(line)
        if match:
            if current_id is not None:
                rows.append((current_id, "\n".join(current_text).strip()))
            current_id = match.group(1)
            remainder = match.group(2)
            current_text = [remainder] if remainder else []
        elif current_id is not None:
            current_text.append(line)

    if current_id is not None:
        rows.append((current_id, "\n".join(current_text).strip()))

    return pd.DataFrame(rows, columns=["id", "text"])


def robust_read_tsv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    except Exception:
        return _parse_raw_tsv(path)


def save_submission(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    output = df[["id", "text"]].copy()
    output.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
