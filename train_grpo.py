#!/usr/bin/env python3
"""Train a code model with GRPO and execution-based HumanEval rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_grpo.io_utils import read_jsonl
from code_grpo.metadata import experiment_metadata
from reward_function import (
    conciseness_reward,
    format_reward,
    make_correctness_reward,
    syntax_reward,
)
from sandbox.executor import LocalExecutor


def validate_dataset(path: str) -> dict:
    rows = list(read_jsonl(path))
    required = {"task_id", "prompt", "starter_code", "test", "entry_point"}
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} missing: {', '.join(sorted(missing))}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task IDs in training data")
    return {"rows": len(rows), "columns": sorted(rows[0]), "path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="data/humaneval_train.jsonl")
    parser.add_argument("--eval-data", default="data/humaneval_validation.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/qwen-code-grpo")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--executor", choices=("local", "docker"), default="local")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--quantization", choices=("none", "4bit"), default="4bit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-local-code-execution", action="store_true")
    args = parser.parse_args()

    train_audit = validate_dataset(args.train_data)
    eval_audit = validate_dataset(args.eval_data)
    audit = {
        "train": train_audit,
        "validation": eval_audit,
        "configuration": vars(args),
        "environment": experiment_metadata(),
    }
    print(json.dumps(audit, indent=2))
    if args.dry_run:
        return
    if args.executor == "local" and not args.allow_local_code_execution:
        raise SystemExit(
            "GRPO correctness rewards execute generated Python. Use an isolated machine/container "
            "and explicitly pass --allow-local-code-execution."
        )

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

    train_dataset = Dataset.from_json(args.train_data)
    eval_dataset = Dataset.from_json(args.eval_data)
    for name in ("canonical_solution",):
        if name in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns(name)
        if name in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns(name)

    effective_batch = args.gradient_accumulation_steps
    if effective_batch % args.num_generations:
        raise ValueError(
            "With one process and batch size 1, gradient_accumulation_steps must be "
            "divisible by num_generations."
        )

    eval_generations = min(2, args.num_generations)
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=eval_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        num_generations_eval=eval_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        reward_weights=[0.80, 0.10, 0.05, 0.05],
        loss_type="dr_grpo",
        scale_rewards=False,
        beta=0.0,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=25,
        eval_strategy="steps",
        eval_steps=25,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    quantization_config = None
    if args.quantization == "4bit":
        if not torch.cuda.is_available():
            raise SystemExit("4-bit QLoRA training requires a supported accelerator")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    if args.executor == "docker":
        from sandbox.docker_runner import DockerExecutor

        executor = DockerExecutor(timeout_seconds=max(args.timeout, 5.0))
    else:
        executor = LocalExecutor(timeout_seconds=args.timeout)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[
            make_correctness_reward(executor),
            syntax_reward,
            format_reward,
            conciseness_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    Path(args.output_dir, "experiment_config.json").write_text(
        json.dumps({"configuration": vars(args), "environment": experiment_metadata()}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
