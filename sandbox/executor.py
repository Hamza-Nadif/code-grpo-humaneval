"""Resource-limited subprocess execution for trusted research environments.

This is defense in depth, not a security boundary. Use the Docker runner for
untrusted model output from unknown sources.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    status: str
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _limit_resources(cpu_seconds: int, memory_mb: int, file_mb: int) -> None:
    memory_bytes = memory_mb * 1024 * 1024
    file_bytes = file_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    os.setsid()


class LocalExecutor:
    """Run one Python candidate in an isolated temporary directory."""

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        memory_mb: int = 512,
        file_mb: int = 8,
        output_chars: int = 4_000,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.file_mb = file_mb
        self.output_chars = output_chars

    def run(self, source: str) -> ExecutionResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="code-grpo-") as temp_dir:
            script = Path(temp_dir) / "candidate.py"
            script.write_text(source, encoding="utf-8")
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": temp_dir,
                "TMPDIR": temp_dir,
            }
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    cwd=temp_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    preexec_fn=lambda: _limit_resources(
                        max(1, int(self.timeout_seconds)), self.memory_mb, self.file_mb
                    ),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    passed=False,
                    status="timeout",
                    duration_seconds=time.monotonic() - started,
                    stdout=(exc.stdout or "")[-self.output_chars :],
                    stderr=(exc.stderr or "")[-self.output_chars :],
                )

        stderr = completed.stderr[-self.output_chars :]
        stdout = completed.stdout[-self.output_chars :]
        if completed.returncode == 0:
            status = "passed"
        elif "AssertionError" in stderr:
            status = "failed_tests"
        elif "SyntaxError" in stderr or "IndentationError" in stderr:
            status = "syntax_error"
        elif completed.returncode < 0:
            status = "killed"
        else:
            status = "runtime_error"
        return ExecutionResult(
            passed=completed.returncode == 0,
            status=status,
            duration_seconds=time.monotonic() - started,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
        )
