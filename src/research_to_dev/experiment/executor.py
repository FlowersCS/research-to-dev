"""CodeExecutor — runs ``run_command`` from program.md via subprocess (CE-01–CE-04).

Executes the shell command defined in program.md frontmatter (D3) with a
configurable timeout.  Captures stdout/stderr and returns an
``ExecutionResult``.  Timeouts are caught and returned as
``timed_out=True`` rather than raised (T1).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of a command execution via ``subprocess.run``.

    Attributes:
        exit_code: Process exit code (0 = success).
        stdout: Captured standard output text.
        stderr: Captured standard error text.
        timed_out: Whether the command exceeded the timeout.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CodeExecutor:
    """Executes a shell command with a configurable timeout (CE-01–CE-04).

    Constructor-injectable so tests can supply an arbitrary working
    directory without touching the real filesystem.

    Usage::

        executor = CodeExecutor(cwd=".")
        result = executor.execute(
            run_command="python train.py --epochs 10",
            timeout=300,
        )
    """

    def __init__(self, *, cwd: str = ".") -> None:
        self._cwd = Path(cwd).resolve()

    def execute(self, run_command: str, timeout: int) -> ExecutionResult:
        """Run a shell command and capture its output.

        Args:
            run_command: Shell command string from program.md frontmatter (D3).
            timeout: Maximum seconds to wait before timing out.

        Returns:
            ExecutionResult with exit code, stdout, stderr, and timeout flag.
            Non-zero exit codes are NOT raised — the caller decides how to
            handle them.
        """
        try:
            result = subprocess.run(
                run_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._cwd,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="",
                timed_out=True,
            )

        return ExecutionResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )
