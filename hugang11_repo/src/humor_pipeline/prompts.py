from __future__ import annotations

from typing import Tuple

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

NEWS_INSTRUCTION = "你是一个幽默的新闻评论员，请对以下新闻进行神吐槽。"
WORDS_INSTRUCTION = "你是一个脑洞大开的脱口秀演员，请用这两个词造一个荒谬的段子。"

STRICT_CONSTRAINT = (
    "你必须严格按照以下格式输出，禁止包含其他多余内容：\n\n"
    "【思考逻辑】\n"
    "(在这里写下你的分析过程，构思多个笑话)\n\n"
    "【神吐槽】\n"
    "(在这里输出你挑选出的唯一的、最好笑的那个段子)"
)


def build_instruction_and_input(row: dict, strict: bool = False) -> Tuple[str, str]:
    headline = str(row.get("headline", "") or "").strip()
    word1 = str(row.get("word1", "") or "").strip()
    word2 = str(row.get("word2", "") or "").strip()

    if headline and headline != "-":
        instruction = NEWS_INSTRUCTION
        user_input = f"新闻标题：{headline}"
    elif word1 and word1 != "-" and word2 and word2 != "-":
        instruction = WORDS_INSTRUCTION
        user_input = f"词汇对：{word1} {word2}"
    else:
        raise ValueError("Row does not contain a valid headline or word pair.")

    if strict:
        instruction = f"{instruction}\n{STRICT_CONSTRAINT}"
    return instruction, user_input


def format_prompt(instruction: str, user_input: str, output: str = "") -> str:
    return ALPACA_PROMPT.format(instruction, user_input, output)
