import json
from pathlib import Path

import pytest

NOTEBOOK = Path("notebooks/code_grpo_humaneval_colab.ipynb")


def test_colab_notebook_is_valid_and_has_required_stages():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    contents = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for required in (
        "build_training_data.py",
        "baseline-heldout",
        "train_grpo.py",
        "RUN_MAIN_TRAINING = False",
        "grpo-heldout",
    ):
        assert required in contents


@pytest.mark.parametrize("steps", [10, 50])
def test_controlled_experiment_notebook(steps):
    path = Path(f"notebooks/code_grpo_humaneval_{steps}_step_experiment.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    contents = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert f"'--max-steps', '{steps}'" in contents
    assert "baseline_pass_at_1" in contents
    assert "trained_pass_at_1" in contents
    assert "colab_files.download" in contents


def test_improved_experiment_addresses_observed_bottlenecks():
    path = Path("notebooks/code_grpo_humaneval_50_step_improved_experiment.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    contents = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "'--num-generations', '4'" in contents
    assert "'--gradient-accumulation-steps', '4'" in contents
    assert "'--max-completion-length', '256'" in contents
    assert contents.count("'--max-new-tokens', '256'") == 2
    assert "zero_gradient_steps" in contents
    assert "mean_training_clipped_ratio" in contents


def test_checkpoint_recovery_notebook_does_not_retrain():
    path = Path("notebooks/evaluate_checkpoint_25_from_archive.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    contents = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "checkpoint-25" in contents
    assert "evaluate_baseline.py" in contents
    assert "train_grpo.py" not in contents
    assert "files.upload()" in contents
    assert "files.download(result_archive)" in contents


def test_sft_grpo_notebook_keeps_test_split_for_evaluation_only():
    path = Path("notebooks/code_sft_grpo_humaneval_experiment.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    contents = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "train_sft.py" in contents
    assert "'--adapter', SFT_DIR" in contents
    assert "base_pass_at_1" in contents
    assert "sft_pass_at_1" in contents
    assert "sft_grpo_pass_at_1" in contents
    assert contents.count("data/humaneval_test.jsonl") == 1
    assert "copy_final_adapter" in contents
