from sandbox.executor import LocalExecutor


def test_executor_reports_success():
    result = LocalExecutor(timeout_seconds=1.0).run("assert 2 + 2 == 4\n")
    assert result.passed
    assert result.status == "passed"


def test_executor_reports_assertion_failure():
    result = LocalExecutor(timeout_seconds=1.0).run("assert False\n")
    assert not result.passed
    assert result.status == "failed_tests"


def test_executor_times_out():
    result = LocalExecutor(timeout_seconds=0.2).run("while True:\n    pass\n")
    assert not result.passed
    assert result.status in {"timeout", "killed"}
