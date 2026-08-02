#!/usr/bin/env python3
"""Warm up a code model with completion-only SFT before GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_grpo.io_utils import read_jsonl
from code_grpo.metadata import experiment_metadata
from code_grpo.prompts import supervised_example
from train_grpo import cast_trainable_parameters_to_fp32, resolve_precision


def validate_sft_dataset(path: str) -> dict:
    rows = list(read_jsonl(path))
    required = {"task_id", "prompt", "canonical_solution"}
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} missing: {', '.join(sorted(missing))}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task IDs in SFT data")
    return {"rows": len(rows), "columns": sorted(rows[0]), "path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="data/humaneval_train.jsonl")
    parser.add_argument("--eval-data", default="data/humaneval_validation.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--output-dir", default="outputs/qwen-code-sft")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--quantization", choices=("none", "4bit"), default="4bit")
    parser.add_argument(
        "--precision", choices=("auto", "fp16", "bf16", "fp32"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audit = {
        "train": validate_sft_dataset(args.train_data),
        "validation": validate_sft_dataset(args.eval_data),
        "configuration": vars(args),
        "environment": experiment_metadata(),
    }
    print(json.dumps(audit, indent=2))
    if args.dry_run:
        return

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

    train_source = Dataset.from_json(args.train_data)
    train_dataset = train_source.map(supervised_example, remove_columns=train_source.column_names)
    eval_source = Dataset.from_json(args.eval_data)
    eval_dataset = eval_source.map(supervised_example, remove_columns=eval_source.column_names)

    precision = resolve_precision(
        args.precision,
        cuda_available=torch.cuda.is_available(),
        bf16_supported=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    model_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]
    print(f"Resolved training precision: {precision}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        completion_only_loss=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=15,
        eval_strategy="steps",
        eval_steps=15,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        model_init_kwargs={"dtype": model_dtype},
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
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
        )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    trainable_dtypes = cast_trainable_parameters_to_fp32(trainer.model)
    print(f"Trainable parameter dtypes after QLoRA preparation: {trainable_dtypes}")
    trainer.train()
    trainer.save_model(args.output_dir)
    Path(args.output_dir, "experiment_config.json").write_text(
        json.dumps(
            {
                "configuration": {**vars(args), "resolved_precision": precision},
                "environment": experiment_metadata(),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
