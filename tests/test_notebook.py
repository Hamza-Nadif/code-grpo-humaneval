import json
from pathlib import Path

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
