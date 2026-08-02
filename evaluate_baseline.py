#!/usr/bin/env python3
"""Generate and evaluate HumanEval completions with auditable pass@k metrics."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from code_grpo.io_utils import read_jsonl, write_jsonl
from code_grpo.metadata import experiment_metadata
from code_grpo.prompts import conversation_prompt
from reward_function import score_completion
from sandbox.executor import LocalExecutor


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased pass@k estimator from the HumanEval paper."""
    if n < k:
        raise ValueError(f"pass@{k} requires at least {k} samples, got {n}")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - correct + 1, n + 1))


def load_model(model_id: str, quantization: str, adapter: str | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit("Model evaluation requires packages from requirements.txt") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"device_map": "auto", "dtype": "auto"}
    if quantization == "4bit":
        if not torch.cuda.is_available():
            raise SystemExit("4-bit model evaluation requires a supported CUDA accelerator")
        model_kwargs["dtype"] = torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Adapter evaluation requires peft") from exc
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generation_kwargs(tokenizer, count: int, args) -> dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "num_return_sequences": count,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
        )
    return kwargs


def generate_completions(model, tokenizer, prompt: str, count: int, args) -> list[str]:
    import torch

    messages = conversation_prompt(prompt)
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = f"You are an expert Python programmer.\n\n{prompt}\n\nAnswer:\n"
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs(tokenizer, count, args),
        )
    prompt_tokens = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(outputs[:, prompt_tokens:], skip_special_tokens=True)


def summarize(scored_rows: list[dict], requested_k: list[int]) -> dict:
    by_task = defaultdict(list)
    statuses = defaultdict(int)
    for row in scored_rows:
        by_task[row["task_id"]].append(bool(row["passed"]))
        statuses[row["status"]] += 1

    metrics = {}
    for k in requested_k:
        eligible = [values for values in by_task.values() if len(values) >= k]
        if eligible:
            total = sum(pass_at_k(len(values), sum(values), k) for values in eligible)
            metrics[f"pass@{k}"] = total / len(eligible)
    return {
        "tasks": len(by_task),
        "samples": len(scored_rows),
        "metrics": metrics,
        "execution_statuses": dict(sorted(statuses.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/humaneval_test.jsonl")
    parser.add_argument(
        "--backend", choices=("transformers", "oracle", "samples"), default="oracle"
    )
    parser.add_argument("--samples-file", help="JSONL with task_id and completion")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--adapter", help="Optional trained LoRA/QLoRA adapter directory")
    parser.add_argument("--quantization", choices=("none", "4bit"), default="none")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", default="1", help="Comma-separated pass@k values")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--executor", choices=("local", "docker"), default="local")
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--allow-local-code-execution", action="store_true")
    args = parser.parse_args()

    if args.executor == "local" and not args.allow_local_code_execution:
        raise SystemExit(
            "Refusing to execute generated code. Re-run only in an isolated environment "
            "with --allow-local-code-execution, or use the Docker runner documented in README.md."
        )
    if args.backend == "samples" and not args.samples_file:
        parser.error("--samples-file is required with --backend samples")

    random.seed(args.seed)
    problems = list(read_jsonl(args.data))
    if args.limit:
        problems = problems[: args.limit]
    problem_by_id = {row["task_id"]: row for row in problems}

    generated_rows = []
    if args.backend == "oracle":
        for row in problems:
            generated_rows.append(
                {"task_id": row["task_id"], "completion": row["canonical_solution"]}
            )
    elif args.backend == "samples":
        generated_rows = [
            row for row in read_jsonl(args.samples_file) if row.get("task_id") in problem_by_id
        ]
    else:
        model, tokenizer = load_model(args.model, args.quantization, args.adapter)
        for index, row in enumerate(problems, start=1):
            completions = generate_completions(
                model, tokenizer, row["prompt"], args.samples_per_task, args
            )
            generated_rows.extend({"task_id": row["task_id"], "completion": c} for c in completions)
            print(f"Generated {index}/{len(problems)} tasks", flush=True)

    if args.executor == "docker":
        from sandbox.docker_runner import DockerExecutor

        executor = DockerExecutor(timeout_seconds=max(args.timeout, 5.0))
    else:
        executor = LocalExecutor(timeout_seconds=args.timeout)
    scored_rows = []
    for sample in generated_rows:
        problem = problem_by_id[sample["task_id"]]
        score = score_completion(
            sample["completion"], problem["starter_code"], problem["test"],
            problem["entry_point"], executor,
        )
        scored_rows.append({**sample, **score})

    requested_k = sorted({int(value) for value in args.k.split(",") if value.strip()})
    summary = summarize(scored_rows, requested_k)
    summary.update({
        "backend": args.backend,
        "model": args.model if args.backend == "transformers" else None,
        "adapter": args.adapter if args.backend == "transformers" else None,
        "data": args.data,
        "warning": "Oracle results validate the harness and are not model benchmark results."
        if args.backend == "oracle" else None,
        "environment": experiment_metadata(),
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "samples.jsonl", generated_rows)
    write_jsonl(output_dir / "scored_samples.jsonl", scored_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
