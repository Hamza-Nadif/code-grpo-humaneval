#!/usr/bin/env python3
"""Build deterministic, leakage-aware HumanEval splits for GRPO experiments."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import urllib.request
from pathlib import Path

from code_grpo.code_utils import make_instruction
from code_grpo.io_utils import sha256_file, write_jsonl

DEFAULT_REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"


def load_humaneval(revision: str) -> tuple[list[dict], str]:
    """Load the small official dataset file from a pinned OpenAI revision."""
    url = f"https://raw.githubusercontent.com/openai/human-eval/{revision}/data/HumanEval.jsonl.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "code-grpo-humaneval/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except OSError as exc:
        raise SystemExit(f"Unable to download the official HumanEval data: {exc}") from exc
    text = gzip.decompress(payload).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()], url


def normalize_problem(problem: dict) -> dict:
    required = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
    missing = sorted(required - problem.keys())
    if missing:
        raise ValueError(f"HumanEval row is missing fields: {', '.join(missing)}")
    return {
        "task_id": problem["task_id"],
        "prompt": make_instruction(problem["prompt"], problem["entry_point"]),
        "starter_code": problem["prompt"],
        "canonical_solution": problem["canonical_solution"],
        "test": problem["test"],
        "entry_point": problem["entry_point"],
    }


def split_rows(
    rows: list[dict], seed: int, train_size: int, validation_size: int
) -> dict[str, list[dict]]:
    if train_size <= 0 or validation_size <= 0:
        raise ValueError("train_size and validation_size must be positive")
    if train_size + validation_size >= len(rows):
        raise ValueError("The requested split leaves no held-out test examples")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": shuffled[:train_size],
        "validation": shuffled[train_size : train_size + validation_size],
        "test": shuffled[train_size + validation_size :],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION, help="Immutable openai/human-eval revision"
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=120)
    parser.add_argument("--validation-size", type=int, default=22)
    args = parser.parse_args()

    raw_rows, source_url = load_humaneval(args.revision)
    rows = [normalize_problem(row) for row in raw_rows]
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate HumanEval task IDs detected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_rows(rows, args.seed, args.train_size, args.validation_size)

    files = {}
    full_path = output_dir / "humaneval_official_full.jsonl"
    write_jsonl(full_path, rows)
    files["official_full"] = {
        "path": str(full_path),
        "rows": len(rows),
        "sha256": sha256_file(full_path),
        "task_ids": [row["task_id"] for row in rows],
    }
    for split_name, split_rows_ in splits.items():
        destination = output_dir / f"humaneval_{split_name}.jsonl"
        write_jsonl(destination, split_rows_)
        files[split_name] = {
            "path": str(destination),
            "rows": len(split_rows_),
            "sha256": sha256_file(destination),
            "task_ids": [row["task_id"] for row in split_rows_],
        }

    manifest = {
        "dataset_id": "openai/human-eval",
        "source_url": source_url,
        "revision": args.revision,
        "seed": args.seed,
        "total_rows": len(rows),
        "warning": (
            "These are internal HumanEval splits. Results are not comparable to the "
            "official 164-task HumanEval benchmark after training on the train split."
        ),
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = {
        "manifest": str(manifest_path),
        "sizes": {key: len(value) for key, value in splits.items()},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
