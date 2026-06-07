"""CodingAgent Protocol and OpenCodeAdapter — swappable interface for
coding agents that modify code (D2, AD-01).

The ``CodingAgent`` Protocol defines the contract; ``OpenCodeAdapter`` is
the MVP implementation that invokes the ``opencode`` CLI.  Prompt is an
absolute path to ``program.md`` (D7), and the model comes from frontmatter
``coding_agent_model`` (D12).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class AgentResult:
    """Result of a coding agent invocation.

    Attributes:
        exit_code: Process exit code (0 = success).
        stdout: Captured standard output text.
        stderr: Captured standard error text.
        files_changed: List of files modified by the agent.
        timed_out: Whether the agent invocation timed out.
    """

    exit_code: int
    stdout: str
    stderr: str
    files_changed: list[Path] = field(default_factory=list)
    timed_out: bool = False


class CodingAgent(Protocol):
    """Protocol for swappable coding agent adapters (D2, AD-01).

    Implementations receive an absolute path to ``program.md`` as the
    prompt (D7) and return an ``AgentResult`` after the agent completes.
    """

    def run(
        self,
        prompt: str,
        working_dir: Path,
        timeout: int,
        model: str,
    ) -> AgentResult:
        """Run the coding agent with the given prompt path.

        Args:
            prompt: Absolute path to ``program.md`` (D7).
            working_dir: Directory where the agent should operate.
            timeout: Maximum seconds to wait for the agent.
            model: Model identifier for the agent (D12).

        Returns:
            AgentResult with exit code, captured output, and timeout flag.
        """
        ...


class OpenCodeAdapter:
    """Invokes the ``opencode`` CLI as a coding agent (AA-02–AA-04, D7, D12).

    Constructor-injectable so tests can supply the binary path
    without relying on PATH.

    Usage::

        adapter = OpenCodeAdapter(binary="opencode")
        result = adapter.run(
            prompt="/path/to/program.md",
            working_dir=Path.cwd(),
            timeout=300,
            model="opencode-go/deepseek-v4-pro",
        )
    """

    def __init__(self, *, binary: str = "opencode") -> None:
        self._binary = binary

    def run(
        self,
        prompt: str,
        working_dir: Path,
        timeout: int,
        model: str,
    ) -> AgentResult:
        """Invoke ``opencode run`` with absolute program.md path (AA-03, D7).

        Args:
            prompt: Absolute path to ``program.md`` (D7).
            working_dir: Directory where the agent should operate.
            timeout: Maximum seconds to wait for the agent.
            model: Model identifier passed via ``--model`` (D12).

        Returns:
            AgentResult with exit code, stdout, stderr, and timeout flag.

        Raises:
            RuntimeError: If the ``opencode`` binary is not found.
        """
        cmd = [
            self._binary,
            "run",
            prompt,
            "--model",
            model,
            "--format",
            "json",
            "--dangerously-skip-permissions",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"'{self._binary}' command not found. "
                f"Make sure opencode is installed and available on PATH."
            ) from None
        except subprocess.TimeoutExpired:
            return AgentResult(
                exit_code=-1,
                stdout="",
                stderr="",
                files_changed=[],
                timed_out=True,
            )

        return AgentResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            files_changed=[],
            timed_out=False,
        )
