"""Shared conversational prompts for training and evaluation."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Return only valid Python code, "
    "without Markdown fences or explanation."
)


def conversation_prompt(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def supervised_example(row: dict) -> dict:
    """Convert a HumanEval row to a conversational prompt-completion pair."""
    return {
        "prompt": conversation_prompt(row["prompt"]),
        "completion": [{"role": "assistant", "content": row["canonical_solution"]}],
    }
