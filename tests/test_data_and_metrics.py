import pytest

from build_training_data import split_rows
from evaluate_baseline import pass_at_k, summarize


def test_split_is_deterministic_and_disjoint():
    rows = [{"task_id": f"task-{index}"} for index in range(10)]
    first = split_rows(rows, seed=42, train_size=6, validation_size=2)
    second = split_rows(rows, seed=42, train_size=6, validation_size=2)
    assert first == second
    ids = [{row["task_id"] for row in first[name]} for name in first]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_split_requires_heldout_examples():
    with pytest.raises(ValueError):
        split_rows([{"task_id": str(i)} for i in range(4)], 42, 3, 1)


def test_pass_at_k_extremes():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(2, 1, 1) == pytest.approx(0.5)


def test_summary_uses_per_task_samples():
    rows = [
        {"task_id": "a", "passed": True, "status": "passed"},
        {"task_id": "a", "passed": False, "status": "failed_tests"},
        {"task_id": "b", "passed": False, "status": "failed_tests"},
        {"task_id": "b", "passed": False, "status": "failed_tests"},
    ]
    summary = summarize(rows, [1, 2])
    assert summary["metrics"]["pass@1"] == pytest.approx(0.25)
    assert summary["metrics"]["pass@2"] == pytest.approx(0.5)
