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
