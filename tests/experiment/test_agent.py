"""Tests for CodingAgent Protocol, AgentResult, and OpenCodeAdapter.

Covers AA-01 through AA-04: Protocol definition, OpenCodeAdapter
invocation (mocked subprocess), missing binary, success, timeout,
and non-zero exit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from research_to_dev.experiment.agent import AgentResult, CodingAgent, OpenCodeAdapter


# ------------------------------------------------------------------
# Protocol definition tests
# ------------------------------------------------------------------


class TestCodingAgentProtocol:
    """AA-01: CodingAgent Protocol is valid and checkable."""

    def test_open_code_adapter_conforms_to_protocol(self) -> None:
        """OpenCodeAdapter structurally matches CodingAgent Protocol."""
        adapter = OpenCodeAdapter(binary="opencode")
        # No isinstance check for Protocol, but the type checker verifies
        # structural compatibility. At runtime, we verify the method exists.
        assert hasattr(adapter, "run")
        assert callable(adapter.run)

    def test_agent_result_dataclass_fields(self) -> None:
        """AgentResult has the required fields with correct defaults."""
        result = AgentResult(exit_code=0, stdout="ok", stderr="")

        assert result.exit_code == 0
        assert result.stdout == "ok"
        assert result.stderr == ""
        assert result.files_changed == []
        assert result.timed_out is False

    def test_agent_result_defaults(self) -> None:
        """AgentResult fields_changed and timed_out have sensible defaults."""
        result = AgentResult(exit_code=0, stdout="", stderr="")

        assert isinstance(result.files_changed, list)
        assert result.timed_out is False

    def test_agent_result_can_set_files_changed(self) -> None:
        """AgentResult files_changed can be populated."""
        result = AgentResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            files_changed=[Path("src/model.py")],
        )
        assert result.files_changed == [Path("src/model.py")]

    def test_agent_result_timed_out_flag(self) -> None:
        """AgentResult timed_out flag is settable."""
        result = AgentResult(
            exit_code=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )
        assert result.timed_out is True


# ------------------------------------------------------------------
# OpenCodeAdapter tests (mocked subprocess)
# ------------------------------------------------------------------


class TestOpenCodeAdapterSuccess:
    """AA-03: OpenCodeAdapter runs opencode and returns AgentResult on success."""

    def test_invokes_opencode_with_correct_args(self) -> None:
        """OpenCodeAdapter passes the prompt path, model, and flags correctly."""
        working_dir = Path("/tmp/test")
        prompt_path = "/path/to/program.md"
        model = "opencode-go/deepseek-v4-pro"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="agent output",
                stderr="",
            )

            adapter = OpenCodeAdapter(binary="opencode")
            result = adapter.run(
                prompt=prompt_path,
                working_dir=working_dir,
                timeout=300,
                model=model,
            )

            # Verify the correct command was constructed
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "opencode"
            assert call_args[1] == "run"
            assert call_args[2] == prompt_path
            assert call_args[3] == "--model"
            assert call_args[4] == model
            assert "--format" in call_args
            assert "json" in call_args
            assert "--dangerously-skip-permissions" in call_args

            # Verify kwargs
            assert mock_run.call_args[1]["timeout"] == 300
            assert mock_run.call_args[1]["cwd"] == working_dir

            # Verify result
            assert isinstance(result, AgentResult)
            assert result.exit_code == 0
            assert result.stdout == "agent output"
            assert result.stderr == ""
            assert result.timed_out is False

    def test_returns_agent_result_on_exit_zero(self) -> None:
        """Success returns AgentResult with exit_code=0 and captured output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="result: accuracy 0.92",
                stderr="some warnings",
            )

            adapter = OpenCodeAdapter()
            result = adapter.run(
                prompt="/tmp/program.md",
                working_dir=Path("/tmp"),
                timeout=60,
                model="test-model",
            )

            assert result.exit_code == 0
            assert result.stdout == "result: accuracy 0.92"
            assert result.stderr == "some warnings"
            assert result.timed_out is False

    def test_returns_agent_result_on_non_zero_exit(self) -> None:
        """Non-zero exit returns AgentResult with the actual exit code."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="partial output",
                stderr="error: something went wrong",
            )

            adapter = OpenCodeAdapter()
            result = adapter.run(
                prompt="/tmp/program.md",
                working_dir=Path("/tmp"),
                timeout=60,
                model="test-model",
            )

            assert result.exit_code == 1
            assert result.stdout == "partial output"
            assert result.stderr == "error: something went wrong"
            assert result.timed_out is False


class TestOpenCodeAdapterMissingBinary:
    """AA-04: Missing opencode binary raises RuntimeError."""

    def test_missing_binary_raises_runtime_error(self) -> None:
        """FileNotFoundError from subprocess → RuntimeError with clear message."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("No such file")

            adapter = OpenCodeAdapter(binary="nonexistent-opencode")

            with pytest.raises(RuntimeError, match="not found"):
                adapter.run(
                    prompt="/tmp/program.md",
                    working_dir=Path("/tmp"),
                    timeout=60,
                    model="test-model",
                )

    def test_missing_binary_message_includes_binary_name(self) -> None:
        """Error message names the missing binary for debugging."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            adapter = OpenCodeAdapter(binary="my-custom-binary")

            with pytest.raises(RuntimeError, match="my-custom-binary"):
                adapter.run(
                    prompt="/tmp/program.md",
                    working_dir=Path("/tmp"),
                    timeout=60,
                    model="test-model",
                )


class TestOpenCodeAdapterTimeout:
    """T1: Timeout returns AgentResult with timed_out=True (not raised)."""

    def test_timeout_returns_timed_out_result(self) -> None:
        """subprocess.TimeoutExpired → AgentResult with timed_out=True."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["opencode", "run", "..."] ,
                timeout=10,
            )

            adapter = OpenCodeAdapter()
            result = adapter.run(
                prompt="/tmp/program.md",
                working_dir=Path("/tmp"),
                timeout=10,
                model="test-model",
            )

            assert result.timed_out is True
            assert result.exit_code == -1
            assert result.stdout == ""
            assert result.stderr == ""

    def test_timeout_does_not_raise(self) -> None:
        """Timeout is caught, not propagated to caller."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["opencode"],
                timeout=5,
            )

            adapter = OpenCodeAdapter()
            # Should NOT raise
            result = adapter.run(
                prompt="/tmp/program.md",
                working_dir=Path("/tmp"),
                timeout=5,
                model="test-model",
            )

            assert result.timed_out is True


class TestOpenCodeAdapterConstructor:
    """Constructor injection for binary path."""

    def test_default_binary_is_opencode(self) -> None:
        """Default constructor uses 'opencode' as the binary."""
        adapter = OpenCodeAdapter()
        assert adapter._binary == "opencode"

    def test_custom_binary_path(self) -> None:
        """Constructor accepts custom binary path."""
        adapter = OpenCodeAdapter(binary="/usr/local/bin/opencode")
        assert adapter._binary == "/usr/local/bin/opencode"

    def test_constructor_is_keyword_only(self) -> None:
        """Constructor uses keyword-only injection pattern."""
        adapter = OpenCodeAdapter(binary="my-opencode")
        assert adapter._binary == "my-opencode"
