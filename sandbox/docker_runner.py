"""Run a candidate with Docker's process, network, and filesystem controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from sandbox.executor import ExecutionResult


class DockerExecutor:
    def __init__(self, image: str = "code-grpo-sandbox:latest", timeout_seconds: float = 5.0):
        self.image = image
        self.timeout_seconds = timeout_seconds

    def run(self, source: str) -> ExecutionResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="code-grpo-docker-") as temp_dir:
            script = Path(temp_dir) / "candidate.py"
            script.write_text(source, encoding="utf-8")
            command = [
                "docker", "run", "--rm", "--network=none", "--read-only",
                "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--pids-limit=32", "--memory=512m", "--cpus=1",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "-v", f"{script}:/workspace/candidate.py:ro",
                self.image,
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True,
                    timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(False, "timeout", time.monotonic() - started,
                                       exc.stdout or "", exc.stderr or "")
        return ExecutionResult(
            passed=completed.returncode == 0,
            status="passed" if completed.returncode == 0 else "failed",
            duration_seconds=time.monotonic() - started,
            stdout=completed.stdout[-4000:], stderr=completed.stderr[-4000:],
            returncode=completed.returncode,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Python file containing candidate plus tests")
    parser.add_argument("--image", default="code-grpo-sandbox:latest")
    args = parser.parse_args()
    result = DockerExecutor(args.image).run(Path(args.source).read_text(encoding="utf-8"))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
