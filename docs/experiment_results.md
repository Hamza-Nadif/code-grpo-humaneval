# Experiment Results

## Base vs SFT vs SFT+GRPO

Run date: 2026-08-02  
Code commit: `a9088cc`  
Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`  
Evaluation: deterministic greedy decoding, 256 new tokens, one sample per task

| Stage | Passed | Internal pass@1 | Delta |
|---|---:|---:|---:|
| Frozen base | 9/22 | 0.4091 | - |
| SFT, 30 steps | 9/22 | 0.4091 | +0.0000 |
| SFT + GRPO, 50 steps | 10/22 | 0.4545 | +0.0455 |

SFT changed every held-out completion. It gained `HumanEval/151` and
`HumanEval/62`, but lost `HumanEval/8` and `HumanEval/6`, resulting in no net
pass@1 change. GRPO then changed one completion and converted `HumanEval/161`
from failed tests to passed without losing another previously correct task.

## Training diagnostics

- Best SFT checkpoint: `checkpoint-30`
- Best SFT validation loss: `0.4404`
- GRPO zero-gradient steps: 24/50
- GRPO mean clipped-completion ratio: 0.0000

The supervised warm-up removed the truncation behavior observed in the earlier
direct-GRPO experiment. GRPO still received no relative learning signal in 24
of 50 steps, so reward diversity remains a useful target for future work.

## Interpretation

This run demonstrates a positive result on the repository's internal held-out
split: one additional task passed after SFT+GRPO. The result is small and comes
from a single training seed and only 22 test tasks. It should be treated as a
controlled pilot result, not evidence of a statistically reliable model
improvement or an official HumanEval leaderboard score.

The deterministic 120/22/22 split was fixed before training. Neither SFT nor
GRPO used the 22 internal test tasks. Canonical solutions were available only
to SFT on the 120-task training split; GRPO used execution rewards on that same
training split.

## Reproduction

Run [`code_sft_grpo_humaneval_experiment.ipynb`](../notebooks/code_sft_grpo_humaneval_experiment.ipynb)
on a Colab T4 GPU. The notebook packages the final adapters, scored samples,
environment metadata, and machine-readable comparison in
`code-sft-grpo-humaneval-results.zip`.
