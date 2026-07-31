from reward_function import (
    conciseness_reward,
    format_reward,
    make_correctness_reward,
    score_completion,
    syntax_reward,
)
from sandbox.executor import LocalExecutor

STARTER = 'def add(a, b):\n    """Return the sum."""\n'
TESTS = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0"


def test_correct_solution_gets_full_correctness_reward():
    reward = make_correctness_reward(LocalExecutor(timeout_seconds=1.0))
    assert reward(["    return a + b"], [STARTER], [TESTS], ["add"]) == [1.0]


def test_wrong_solution_fails_hidden_tests():
    reward = make_correctness_reward(LocalExecutor(timeout_seconds=1.0))
    assert reward(["    return a - b"], [STARTER], [TESTS], ["add"]) == [0.0]


def test_reward_components_are_bounded():
    completion = "    return a + b"
    assert syntax_reward([completion], [STARTER], ["add"]) == [1.0]
    assert format_reward([completion], [STARTER], ["add"]) == [1.0]
    assert conciseness_reward([completion]) == [1.0]
    score = score_completion(completion, STARTER, TESTS, "add", LocalExecutor(timeout_seconds=1.0))
    assert score["total"] == 1.0
    assert score["passed"] is True


def test_markdown_is_valid_but_loses_format_reward():
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    assert syntax_reward([completion], [STARTER], ["add"]) == [1.0]
    assert format_reward([completion], [STARTER], ["add"]) == [0.0]
