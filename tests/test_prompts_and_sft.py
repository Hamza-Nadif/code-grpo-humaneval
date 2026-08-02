import json

import pytest

from code_grpo.prompts import SYSTEM_PROMPT, conversation_prompt, supervised_example
from train_sft import validate_sft_dataset


def test_conversation_prompt_matches_supervised_format():
    row = {"prompt": "Complete this function", "canonical_solution": "    return 1\n"}

    example = supervised_example(row)

    assert example["prompt"] == conversation_prompt(row["prompt"])
    assert example["prompt"][0]["content"] == SYSTEM_PROMPT
    assert example["completion"] == [
        {"role": "assistant", "content": row["canonical_solution"]}
    ]


def test_validate_sft_dataset_requires_canonical_solutions(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"task_id": "one", "prompt": "prompt"}) + "\n")

    with pytest.raises(ValueError, match="canonical_solution"):
        validate_sft_dataset(str(path))


def test_validate_sft_dataset_reports_valid_rows(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {"task_id": "one", "prompt": "prompt", "canonical_solution": "return 1"}
    path.write_text(json.dumps(row) + "\n")

    audit = validate_sft_dataset(str(path))

    assert audit["rows"] == 1
    assert audit["path"] == str(path)
