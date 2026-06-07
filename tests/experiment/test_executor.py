"""Tests for CodeExecutor: successful execution, non-zero exit, timeout,
and command failure.

Covers CE-01 through CE-04: subprocess invocation, output capture,
timeout handling, and non-zero exit code passthrough.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from research_to_dev.experiment.executor import CodeExecutor, ExecutionResult


# ------------------------------------------------------------------
# ExecutionResult dataclass tests
# ------------------------------------------------------------------


class TestExecutionResult:
    """ExecutionResult dataclass field validation."""

    def test_default_fields(self) -> None:
        """timed_out defaults to False."""
        result = ExecutionResult(exit_code=0, stdout="ok", stderr="")
        assert result.timed_out is False

    def test_fields_are_accessible(self) -> None:
        """All fields can be set and read."""
        result = ExecutionResult(
            exit_code=1,
            stdout="some output",
            stderr="some error",
            timed_out=True,
        )
        assert result.exit_code == 1
        assert result.stdout == "some output"
        assert result.stderr == "some error"
        assert result.timed_out is True


# ------------------------------------------------------------------
# CodeExecutor.execute() tests (mocked subprocess)
# ------------------------------------------------------------------


class TestCodeExecutorSuccess:
    """CE-01, CE-02: Successful command execution captures output."""

    def test_executes_command_via_subprocess(self) -> None:
        """CodeExecutor delegates to subprocess.run with shell=True."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="accuracy: 0.92\n",
                stderr="",
            )

            executor = CodeExecutor(cwd="/tmp/test")
            result = executor.execute(
                run_command="python train.py --epochs 10",
                timeout=60,
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args

            # Verify shell=True and text=True
            assert call_args[1]["shell"] is True
            assert call_args[1]["capture_output"] is True
            assert call_args[1]["text"] is True
            assert call_args[1]["timeout"] == 60
            assert call_args[1]["cwd"] == Path("/tmp/test")

            # Verify result
            assert isinstance(result, ExecutionResult)
            assert result.exit_code == 0
            assert result.stdout == "accuracy: 0.92\n"
            assert result.stderr == ""
            assert result.timed_out is False

    def test_captures_stdout(self) -> None:
        """CE-02: stdout is captured as text."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="val_loss: 0.42\naccuracy: 0.88\n",
                stderr="",
            )

            executor = CodeExecutor()
            result = executor.execute("python train.py", 30)

            assert "val_loss: 0.42" in result.stdout
            assert "accuracy: 0.88" in result.stdout

    def test_captures_stderr(self) -> None:
        """CE-02: stderr is captured as text."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="done",
                stderr="Warning: deprecated API\n",
            )

            executor = CodeExecutor()
            result = executor.execute("python train.py", 30)

            assert "Warning: deprecated API" in result.stderr

    def test_passes_run_command_verbatim(self) -> None:
        """The run_command string is passed as-is to subprocess.run."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="",
                stderr="",
            )

            cmd = "python src/train.py --lr 0.001 --epochs 50 2>&1 | tee log.txt"
            executor = CodeExecutor()
            executor.execute(run_command=cmd, timeout=120)

            assert mock_run.call_args[0][0] == cmd


class TestCodeExecutorNonZeroExit:
    """CE-04: Non-zero exit code is returned, NOT raised."""

    def test_non_zero_exit_returns_exit_code(self) -> None:
        """Exit code 1 → ExecutionResult with exit_code=1."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="started...",
                stderr="Error: out of memory",
            )

            executor = CodeExecutor()
            result = executor.execute("python train.py", 60)

            assert result.exit_code == 1
            assert result.stdout == "started..."
            assert result.stderr == "Error: out of memory"
            assert result.timed_out is False

    def test_non_zero_exit_does_not_raise(self) -> None:
        """CE-04: Non-zero exit should NOT raise an exception."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=137,  # SIGKILL
                stdout="",
                stderr="Killed",
            )

            executor = CodeExecutor()
            # Should NOT raise
            result = executor.execute("python train.py", 60)

            assert result.exit_code == 137

    def test_exit_code_negative(self) -> None:
        """Negative exit codes (e.g., from signals) are passed through."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=-9,  # SIGKILL on some systems
                stdout="",
                stderr="",
            )

            executor = CodeExecutor()
            result = executor.execute("failing-command", 30)

            assert result.exit_code == -9


class TestCodeExecutorTimeout:
    """CE-03 / T1: Timeout returns ExecutionResult with timed_out=True."""

    def test_timeout_returns_timed_out_result(self) -> None:
        """subprocess.TimeoutExpired → ExecutionResult(timed_out=True)."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="python train.py",
                timeout=30,
            )

            executor = CodeExecutor()
            result = executor.execute("python train.py", 30)

            assert result.timed_out is True
            assert result.exit_code == -1
            assert result.stdout == ""
            assert result.stderr == ""

    def test_timeout_does_not_raise(self) -> None:
        """Timeout is caught internally, not propagated."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="python slow_script.py",
                timeout=5,
            )

            executor = CodeExecutor()
            # Should NOT raise
            result = executor.execute("python slow_script.py", 5)

            assert result.timed_out is True


class TestCodeExecutorConstructor:
    """Constructor injection pattern for cwd."""

    def test_default_cwd_is_current_directory(self) -> None:
        """Default constructor uses '.', resolved to absolute."""
        executor = CodeExecutor()
        assert executor._cwd == Path.cwd()

    def test_custom_cwd(self) -> None:
        """Custom cwd is resolved to absolute path."""
        executor = CodeExecutor(cwd="/tmp/experiments")
        assert executor._cwd == Path("/tmp/experiments")

    def test_keyword_only_constructor(self) -> None:
        """Constructor uses keyword-only injection (existing pattern)."""
        executor = CodeExecutor(cwd="/custom/path")
        assert executor._cwd == Path("/custom/path")
