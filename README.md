# Code GRPO on HumanEval

[Open the one-click smoke test in Google Colab](https://colab.research.google.com/github/Hamza-Nadif/code-grpo-humaneval/blob/main/notebooks/code_grpo_humaneval_colab_one_click.ipynb)

[Run the automatic 10-step before/after experiment in Google Colab](https://colab.research.google.com/github/Hamza-Nadif/code-grpo-humaneval/blob/main/notebooks/code_grpo_humaneval_10_step_experiment.ipynb)

[Run the automatic 50-step follow-up experiment in Google Colab](https://colab.research.google.com/github/Hamza-Nadif/code-grpo-humaneval/blob/main/notebooks/code_grpo_humaneval_50_step_experiment.ipynb)

[Run the improved 50-step diversity experiment in Google Colab](https://colab.research.google.com/github/Hamza-Nadif/code-grpo-humaneval/blob/main/notebooks/code_grpo_humaneval_50_step_improved_experiment.ipynb)

[Evaluate checkpoint 25 from the downloaded experiment archive](https://colab.research.google.com/github/Hamza-Nadif/code-grpo-humaneval/blob/main/notebooks/evaluate_checkpoint_25_from_archive.ipynb)

[Run the Base vs SFT vs SFT+GRPO experiment in Google Colab](https://colab.research.google.com/github/Hamza-Nadif/code-grpo-humaneval/blob/main/notebooks/code_sft_grpo_humaneval_experiment.ipynb)

[Replicate SFT+GRPO with seeds 7 and 123](https://colab.research.google.com/github/Hamza-Nadif/code-grpo-humaneval/blob/main/notebooks/code_sft_grpo_multiseed_replication.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hamza-Nadif/code-grpo-humaneval/blob/main/notebooks/code_grpo_humaneval_colab.ipynb)

A reproducible research project for improving Python code generation with
Group Relative Policy Optimization (GRPO), execution-based rewards, and a
leakage-aware HumanEval protocol.

Latest controlled result: Base `0.4091` -> SFT `0.4091` -> SFT+GRPO `0.4545`
on the 22-task internal held-out split. See the
[experiment report](docs/experiment_results.md) for task-level changes and
limitations.

The main controlled experiment is split into five stages:

1. build deterministic train/validation/test data;
2. measure a frozen model baseline;
3. warm up a QLoRA adapter with completion-only supervised fine-tuning (SFT);
4. continue that adapter with execution-reward GRPO;
5. compare Base, SFT, and SFT+GRPO on the untouched internal test split.

For a hosted GPU run that does not use local disk or GPU resources, open the
Colab notebook with the badge above and execute its cells in order.

> **Security:** model-generated code is untrusted. The local executor adds
> time, memory, process, and file limits, but it is not a security boundary.
> Prefer the Docker executor on a disposable machine. The official HumanEval
> repository gives the same warning about arbitrary code execution.

## Why this design

HumanEval contains 164 handwritten Python tasks with hidden unit tests. GRPO
samples a group of answers for each prompt, scores them, and increases the
relative probability of stronger answers. Here the main reward is functional
correctness, supported by smaller syntax, output-format, and conciseness
rewards:

```text
total = 0.80 * tests_passed
      + 0.10 * valid_python
      + 0.05 * clean_output
      + 0.05 * concise_output
```

Training directly on all 164 HumanEval tests and then reporting official
HumanEval pass@k would be data leakage. `build_training_data.py` therefore
creates a deterministic 120/22/22 internal split. Results after training are
reported as **HumanEval internal held-out**, not as the official benchmark.
The complete 164-task file is also written for the pre-training baseline only.

## Project structure

```text
code-grpo-humaneval/
├── README.md
├── evaluate_baseline.py       # generation, execution, pass@k, JSON reports
├── build_training_data.py     # pinned official data and deterministic splits
├── reward_function.py         # correctness + syntax + format + length rewards
├── train_sft.py               # completion-only SFT/QLoRA warm-up
├── train_grpo.py              # TRL GRPOTrainer with LoRA/QLoRA
├── code_grpo/                 # shared prompts, extraction, and JSONL utilities
├── sandbox/                   # local limits and hardened Docker runner
├── configs/                   # reference smoke configuration
├── tests/                     # unit and integration tests
├── results/                   # generated reports are ignored by Git
└── requirements.txt
```

## Installation

Python 3.10+ is required. Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

For CUDA, install the PyTorch wheel matching the machine first, then install
the remaining requirements. A 4 GB GPU is useful for small inference tests but
is generally too constrained for comfortable GRPO generation and optimization.
The training command is intended for a larger GPU or a cloud notebook.

## 1. Build the data

The dataset is downloaded from a pinned commit of the official OpenAI
HumanEval repository. No model is downloaded here.

```bash
python build_training_data.py --output-dir data
```

Generated files:

- `humaneval_official_full.jsonl`: pre-training reference benchmark;
- `humaneval_train.jsonl`: reward-bearing GRPO tasks;
- `humaneval_validation.jsonl`: model selection;
- `humaneval_test.jsonl`: final internal held-out evaluation;
- `manifest.json`: source revision, checksums, seed, and task IDs.

## 2. Validate the evaluation harness

First build the Docker image:

```bash
docker build -t code-grpo-sandbox:latest sandbox/
```

Then run canonical solutions through five tasks:

```bash
python evaluate_baseline.py \
  --data data/humaneval_test.jsonl \
  --backend oracle \
  --executor docker \
  --limit 5 \
  --output-dir results/oracle-smoke
```

The oracle should obtain `pass@1 = 1.0`. This only validates data assembly,
execution, and metrics; it is not a model score.

For trusted local development without Docker, add both:

```bash
--executor local --allow-local-code-execution
```

## 3. Measure the frozen baseline

Run this **before training** on the complete official data:

```bash
python evaluate_baseline.py \
  --data data/humaneval_official_full.jsonl \
  --backend transformers \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --samples-per-task 1 \
  --executor docker \
  --output-dir results/baseline-official
```

For pass@10, generate at least ten samples per task and set `--k 1,10`.
The script uses the unbiased estimator from the HumanEval paper and refuses to
compute a requested k when too few samples are available.

## 4. Audit training without loading a model

```bash
python train_grpo.py --dry-run
```

This checks required columns, duplicate task IDs, split files, and effective
batch constraints.

## 5. Train directly with GRPO and QLoRA

```bash
python train_grpo.py \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --train-data data/humaneval_train.jsonl \
  --eval-data data/humaneval_validation.jsonl \
  --quantization 4bit \
  --precision auto \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-steps 100 \
  --executor docker \
  --output-dir outputs/qwen-code-grpo
```

Docker startup per completion is intentionally conservative and slow. On a
disposable training host, the local executor is faster but must be enabled
explicitly with `--allow-local-code-execution`.

## 5b. Recommended SFT then GRPO path

Use only the 120-task training split for the supervised warm-up:

```bash
python train_sft.py \
  --train-data data/humaneval_train.jsonl \
  --eval-data data/humaneval_validation.jsonl \
  --quantization 4bit \
  --max-steps 30 \
  --output-dir outputs/qwen-code-sft
```

Then continue the same trainable adapter with execution-reward GRPO:

```bash
python train_grpo.py \
  --adapter outputs/qwen-code-sft \
  --train-data data/humaneval_train.jsonl \
  --eval-data data/humaneval_validation.jsonl \
  --quantization 4bit \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-completion-length 256 \
  --max-steps 50 \
  --executor docker \
  --output-dir outputs/qwen-code-sft-grpo
```

The SFT loss is computed only on canonical completions. Both training stages
exclude the 22-task internal test split.

## 6. Evaluate the trained adapter

```bash
python evaluate_baseline.py \
  --data data/humaneval_test.jsonl \
  --backend transformers \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --adapter outputs/qwen-code-grpo \
  --samples-per-task 1 \
  --executor docker \
  --output-dir results/grpo-heldout
```

Compare `results/baseline-heldout/summary.json` and
`results/grpo-heldout/summary.json` using the same seed, model, generation
settings, sample count, and internal test tasks.

## Tests and quality checks

```bash
pytest
ruff check .
```

The test suite verifies:

- indented HumanEval continuations and complete-function responses;
- Markdown code extraction;
- correct, incorrect, malformed, and timed-out programs;
- deterministic and disjoint dataset splits;
- pass@k estimation and per-task aggregation;
- every component of the GRPO reward.

## Result integrity

Do not report an oracle score as a model result. Do not compare a model trained
on HumanEval tasks with the official 164-task HumanEval leaderboard. Record the
model revision, adapter commit, data manifest hash, generation settings, random
seed, hardware, and package versions for every experiment.

## References

- [OpenAI HumanEval repository](https://github.com/openai/human-eval)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [Hugging Face TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Transformers bitsandbytes and QLoRA guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
