"""Composable rewards for code generation with HumanEval-style tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

from code_grpo.code_utils import (
    build_candidate,
    completion_text,
    extract_code,
    has_expected_function,
    is_valid_python,
)
from sandbox.executor import ExecutionResult, LocalExecutor

Executor = LocalExecutor


def _broadcast(values: object, count: int, name: str) -> list:
    if isinstance(values, str):
        return [values] * count
    if isinstance(values, Sequence):
        if len(values) != count:
            raise ValueError(f"{name} has {len(values)} values for {count} completions")
        return list(values)
    raise TypeError(f"{name} must be a string or sequence")


def syntax_reward(
    completions: Sequence[object],
    starter_code: Sequence[str],
    entry_point: Sequence[str],
    **_: object,
) -> list[float]:
    starters = _broadcast(starter_code, len(completions), "starter_code")
    entries = _broadcast(entry_point, len(completions), "entry_point")
    return [
        1.0 if is_valid_python(build_candidate(s, c, e)) else 0.0
        for c, s, e in zip(completions, starters, entries, strict=True)
    ]


def format_reward(
    completions: Sequence[object],
    starter_code: Sequence[str],
    entry_point: Sequence[str],
    **_: object,
) -> list[float]:
    starters = _broadcast(starter_code, len(completions), "starter_code")
    entries = _broadcast(entry_point, len(completions), "entry_point")
    rewards = []
    for completion, starter, entry in zip(completions, starters, entries, strict=True):
        raw = completion_text(completion).strip()
        candidate = build_candidate(starter, completion, entry)
        clean = bool(raw) and "```" not in raw and raw == extract_code(raw)
        rewards.append(1.0 if clean and has_expected_function(candidate, entry) else 0.0)
    return rewards


def conciseness_reward(completions: Sequence[object], **_: object) -> list[float]:
    """Prefer useful, bounded answers without rewarding empty output."""
    rewards = []
    for completion in completions:
        length = len(extract_code(completion_text(completion)))
        if length < 8:
            reward = 0.0
        elif length <= 1_500:
            reward = 1.0
        else:
            reward = max(0.0, 1.0 - (length - 1_500) / 3_000)
        rewards.append(reward)
    return rewards


def assemble_test_program(
    starter_code: str, completion: object, test: str, entry_point: str
) -> str:
    candidate = build_candidate(starter_code, completion, entry_point)
    return f"{candidate}\n{test.rstrip()}\n\ncheck({entry_point})\n"


def make_correctness_reward(executor: Executor | None = None) -> Callable[..., list[float]]:
    runner = executor or LocalExecutor()

    @lru_cache(maxsize=16_384)
    def execute(source: str) -> ExecutionResult:
        return runner.run(source)

    def correctness_reward(
        completions: Sequence[object],
        starter_code: Sequence[str],
        test: Sequence[str],
        entry_point: Sequence[str],
        **_: object,
    ) -> list[float]:
        count = len(completions)
        starters = _broadcast(starter_code, count, "starter_code")
        tests = _broadcast(test, count, "test")
        entries = _broadcast(entry_point, count, "entry_point")
        rewards = []
        values = zip(completions, starters, tests, entries, strict=True)
        for completion, starter, hidden_test, entry in values:
            source = assemble_test_program(starter, completion, hidden_test, entry)
            if not is_valid_python(source):
                rewards.append(0.0)
                continue
            rewards.append(1.0 if execute(source).passed else 0.0)
        return rewards

    correctness_reward.__name__ = "correctness_reward"
    return correctness_reward


def score_completion(
    completion: object,
    starter_code: str,
    test: str,
    entry_point: str,
    executor: Executor | None = None,
) -> dict[str, float | str | bool]:
    """Return an auditable, offline reward breakdown for one completion."""
    runner = executor or LocalExecutor()
    source = assemble_test_program(starter_code, completion, test, entry_point)
    syntax = float(is_valid_python(source))
    execution = runner.run(source) if syntax else ExecutionResult(False, "syntax_error", 0.0)
    fmt = format_reward([completion], [starter_code], [entry_point])[0]
    concise = conciseness_reward([completion])[0]
    total = 0.8 * float(execution.passed) + 0.1 * syntax + 0.05 * fmt + 0.05 * concise
    return {
        "total": total,
        "correctness": float(execution.passed),
        "syntax": syntax,
        "format": fmt,
        "conciseness": concise,
        "passed": execution.passed,
        "status": execution.status,
        "duration_seconds": execution.duration_seconds,
    }
