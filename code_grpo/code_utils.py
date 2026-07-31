"""Code extraction and assembly helpers shared by training and evaluation."""

from __future__ import annotations

import ast
import re

_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def completion_text(completion: object) -> str:
    """Normalize TRL plain-text and conversational completions to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion or "")


def extract_code(text: str) -> str:
    """Extract Python from a model response while preserving plain completions."""
    # HumanEval canonical solutions are indented continuations. Do not strip
    # leading spaces unless we are removing a prose prefix or Markdown fence.
    text = text.strip("\r\n")
    fenced = _FENCE_RE.findall(text)
    if fenced:
        text = max(fenced, key=len).strip()

    prefixes = ("Here is the code:", "Here's the code:", "Solution:")
    probe = text.lstrip()
    for prefix in prefixes:
        if probe.lower().startswith(prefix.lower()):
            text = probe[len(prefix) :].lstrip()
            break
    return text


def build_candidate(starter_code: str, completion: object, entry_point: str) -> str:
    """Build an executable candidate from a full function or a continuation."""
    code = extract_code(completion_text(completion))
    function_pattern = re.compile(rf"(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(")
    if function_pattern.search(code):
        return code.rstrip() + "\n"
    return starter_code.rstrip() + "\n" + code.rstrip() + "\n"


def is_valid_python(source: str) -> bool:
    """Return whether source parses as Python without executing it."""
    try:
        ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


def has_expected_function(source: str, entry_point: str) -> bool:
    """Check that the candidate defines the expected function."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in ast.walk(tree)
    )


def make_instruction(starter_code: str, entry_point: str) -> str:
    """Create the training/evaluation prompt without exposing hidden tests."""
    return (
        "Complete the Python function below. Return only valid Python code, "
        "without Markdown or explanation. You may return the complete function.\n\n"
        f"{starter_code.rstrip()}\n\n"
        f"The required entry point is `{entry_point}`."
    )
