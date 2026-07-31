from code_grpo.code_utils import (
    build_candidate,
    extract_code,
    has_expected_function,
    is_valid_python,
)

STARTER = 'def add(a, b):\n    """Return the sum."""\n'


def test_preserves_indented_humaneval_continuation():
    candidate = build_candidate(STARTER, "    return a + b\n", "add")
    assert "\n    return a + b\n" in candidate
    assert is_valid_python(candidate)


def test_extracts_fenced_complete_function():
    response = "Explanation\n```python\ndef add(a, b):\n    return a + b\n```"
    candidate = build_candidate(STARTER, response, "add")
    assert candidate.startswith("def add")
    assert candidate.count("def add") == 1
    assert has_expected_function(candidate, "add")


def test_rejects_invalid_python():
    assert not is_valid_python("def broken(:\n")


def test_plain_extraction_does_not_remove_indentation():
    assert extract_code("    return 1\n") == "    return 1"
