"""Experiment setup orchestrator — validates inputs, creates git branch, and writes program.md.

Wires together MetricRegistry, GitOperations, and ProgramWriter following
the Hexagonal Architecture pattern.  All dependencies are constructor-injected.
"""

from __future__ import annotations

import re
from pathlib import Path

from research_to_dev.experiment.git import GitOperations
from research_to_dev.experiment.metrics import MetricRegistry
from research_to_dev.experiment.program import ProgramWriter
from research_to_dev.hypothesis.types import Hypothesis
from research_to_dev.shared.config import ExperimentConfig

# Duration regex: accepts formats like "30m", "2h", "1h30m", "45m", "3h"
_DURATION_RE = re.compile(r"^(\d+h)?(\d+m)$")


class ExperimentSetup:
    """Orchestrates the full experiment setup flow.

    Constructor-injectable dependencies (MetricRegistry, GitOperations,
    ProgramWriter) mean the orchestrator is fully testable with mocks
    for git ops and metrics.

    Usage::

        setup = ExperimentSetup(git_ops=git, writer=writer, metrics=metrics)
        setup.run(hypothesis, config, "accuracy")
    """

    def __init__(
        self,
        *,
        git_ops: GitOperations,
        writer: ProgramWriter,
        metrics: MetricRegistry,
    ) -> None:
        self._git = git_ops
        self._writer = writer
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        hypothesis: Hypothesis,
        config: ExperimentConfig,
        target_metric: str,
    ) -> Path:
        """Run the full experiment setup flow.

        Args:
            hypothesis: The hypothesis to experiment on.
            config: Experiment configuration (time budget, max iterations).
            target_metric: The validated metric name (already resolved).

        Returns:
            Path to the created ``program.md`` file.

        Raises:
            ValueError: If input validation fails (invalid time budget,
                zero/negative max iterations, unknown metric, etc.).
            RuntimeError: If git operations fail (dirty repo, existing
                branch, not a repo).
        """
        # -- 0. Validate inputs --------------------------------------------
        self._validate_time_budget(config.time_budget)
        self._validate_max_iterations(config.max_iterations)

        if not self._metrics.validate_metric(target_metric):
            builtins = ", ".join(self._metrics.list_all())
            raise ValueError(
                f"Target metric '{target_metric}' not found in built-in "
                f"metrics or config.yaml.\n"
                f"Available built-in metrics: {builtins}\n"
                f"Define custom metrics via `research-to-dev config init` "
                f"and edit .research-to-dev/config.yaml"
            )

        # -- 1. Git checks --------------------------------------------------
        if not self._git.is_repo():
            raise RuntimeError(
                "Not a git repository. "
                "Experiments require git for isolation and rollback."
            )

        branch_name = f"experiment/{hypothesis.id}"
        if self._git.branch_exists(branch_name):
            raise RuntimeError(
                f"Experiment branch '{branch_name}' already exists."
            )

        if self._git.is_dirty():
            raise RuntimeError(
                "Working tree has uncommitted changes. "
                "Commit or stash before setting up an experiment."
            )

        # -- 2. Create branch -----------------------------------------------
        self._git.create_branch(branch_name)

        # -- 3. Generate program.md -----------------------------------------
        return self._writer.write(hypothesis, config, target_metric)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_time_budget(time_budget: str) -> None:
        """Validate the time_budget duration string.

        Accepts formats like ``"30m"``, ``"2h"``, ``"1h30m"``.

        Raises:
            ValueError: If the format is invalid.
        """
        if not _DURATION_RE.match(time_budget):
            raise ValueError(
                f"Invalid time budget format '{time_budget}'. "
                f"Use format like '30m', '2h', or '1h30m'."
            )

    @staticmethod
    def _validate_max_iterations(max_iterations: int) -> None:
        """Validate max_iterations is a positive integer.

        Raises:
            ValueError: If ``max_iterations < 1``.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
