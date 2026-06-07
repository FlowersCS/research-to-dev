"""Integration tests for ExperimentSetup orchestrator.

Tests full flow: metric validation, git checks, branch creation,
and program.md generation.  Uses mocks for git ops and metrics
to isolate logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from research_to_dev.experiment.git import GitOperations
from research_to_dev.experiment.metrics import MetricRegistry
from research_to_dev.experiment.program import ProgramWriter
from research_to_dev.experiment.setup import ExperimentSetup
from research_to_dev.hypothesis.types import (
    Hypothesis,
    HypothesisScores,
)
from research_to_dev.shared.config import ExperimentConfig


def _make_hypothesis(**overrides: object) -> Hypothesis:
    defaults: dict[str, object] = {
        "id": "abc123def456",
        "title": "Test hypothesis",
        "description": "A test hypothesis for experiments.",
        "approach": "Implement X",
        "supporting_papers": ["Paper A"],
        "target_metric": "accuracy",
        "expected_improvement": "10% improvement",
        "code_changes": "src/model.py",
        "scores": HypothesisScores(relevance=8, feasibility=7, evidence=6),
        "composite": 7.0,
        "success_criteria": "accuracy > 0.95",
    }
    defaults.update(overrides)
    return Hypothesis(**defaults)  # type: ignore[arg-type]


def _make_config(**overrides: object) -> ExperimentConfig:
    defaults: dict[str, object] = {
        "time_budget": "30m",
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)  # type: ignore[arg-type]


class TestExperimentSetupValidation:
    """Validate inputs before any git or filesystem operations."""

    def test_invalid_time_budget_raises(self) -> None:
        """Invalid time_budget format raises ValueError."""
        git_ops = MagicMock(spec=GitOperations)
        writer = MagicMock(spec=ProgramWriter)
        metrics = MagicMock(spec=MetricRegistry)
        metrics.validate_metric.return_value = True

        setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
        hypothesis = _make_hypothesis()
        config = _make_config(time_budget="not-a-duration")

        with pytest.raises(ValueError, match="Invalid time budget"):
            setup.run(
                hypothesis, config, "accuracy",
                run_command="python train.py",
                baseline={"accuracy": 0.72},
                coding_agent_model="test-model",
            )

    def test_zero_max_iterations_raises(self) -> None:
        """max_iterations = 0 raises ValueError."""
        git_ops = MagicMock(spec=GitOperations)
        writer = MagicMock(spec=ProgramWriter)
        metrics = MagicMock(spec=MetricRegistry)
        metrics.validate_metric.return_value = True

        setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
        hypothesis = _make_hypothesis()
        config = _make_config(max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            setup.run(
                hypothesis, config, "accuracy",
                run_command="python train.py",
                baseline={"accuracy": 0.72},
                coding_agent_model="test-model",
            )

    def test_negative_max_iterations_raises(self) -> None:
        """Negative max_iterations raises ValueError."""
        git_ops = MagicMock(spec=GitOperations)
        writer = MagicMock(spec=ProgramWriter)
        metrics = MagicMock(spec=MetricRegistry)
        metrics.validate_metric.return_value = True

        setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
        hypothesis = _make_hypothesis()
        config = _make_config(max_iterations=-5)

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            setup.run(
                hypothesis, config, "accuracy",
                run_command="python train.py",
                baseline={"accuracy": 0.72},
                coding_agent_model="test-model",
            )

    def test_unknown_metric_raises(self) -> None:
        """Unknown target_metric raises ValueError with actionable message."""
        git_ops = MagicMock(spec=GitOperations)
        writer = MagicMock(spec=ProgramWriter)
        metrics = MetricRegistry()  # real registry

        setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
        hypothesis = _make_hypothesis()
        config = _make_config()

        with pytest.raises(ValueError, match="not found"):
            setup.run(
                hypothesis, config, "not_a_metric",
                run_command="python train.py",
                baseline={"accuracy": 0.72},
                coding_agent_model="test-model",
            )


class TestExperimentSetupGitFailure:
    """Git failures are propagated correctly."""

    def test_dirty_repo_raises_runtime_error(self) -> None:
        """Dirty working tree raises RuntimeError."""
        git_ops = MagicMock(spec=GitOperations)
        git_ops.is_repo.return_value = True
        git_ops.branch_exists.return_value = False
        git_ops.is_dirty.return_value = True

        writer = MagicMock(spec=ProgramWriter)
        metrics = MagicMock(spec=MetricRegistry)
        metrics.validate_metric.return_value = True

        setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
        hypothesis = _make_hypothesis()
        config = _make_config()

        with pytest.raises(RuntimeError, match="uncommitted changes"):
            setup.run(
                hypothesis, config, "accuracy",
                run_command="python train.py",
                baseline={"accuracy": 0.72},
                coding_agent_model="test-model",
            )

    def test_not_a_git_repo_raises_runtime_error(self) -> None:
        """Not inside a git repo raises RuntimeError."""
        git_ops = MagicMock(spec=GitOperations)
        git_ops.is_repo.return_value = False

        writer = MagicMock(spec=ProgramWriter)
        metrics = MagicMock(spec=MetricRegistry)
        metrics.validate_metric.return_value = True

        setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
        hypothesis = _make_hypothesis()
        config = _make_config()

        with pytest.raises(RuntimeError, match="Not a git repository"):
            setup.run(
                hypothesis, config, "accuracy",
                run_command="python train.py",
                baseline={"accuracy": 0.72},
                coding_agent_model="test-model",
            )

    def test_existing_branch_raises_runtime_error(self) -> None:
        """Existing experiment branch raises RuntimeError."""
        git_ops = MagicMock(spec=GitOperations)
        git_ops.is_repo.return_value = True
        git_ops.branch_exists.return_value = True  # already exists!

        writer = MagicMock(spec=ProgramWriter)
        metrics = MagicMock(spec=MetricRegistry)
        metrics.validate_metric.return_value = True

        setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
        hypothesis = _make_hypothesis()
        config = _make_config()

        with pytest.raises(RuntimeError, match="already exists"):
            setup.run(
                hypothesis, config, "accuracy",
                run_command="python train.py",
                baseline={"accuracy": 0.72},
                coding_agent_model="test-model",
            )


class TestExperimentSetupHappyPath:
    """Happy path: all validations pass, branch created, program.md written."""

    def test_happy_path_creates_branch_and_program(self) -> None:
        """On the happy path, branch is created and program.md is written."""
        git_ops = MagicMock(spec=GitOperations)
        git_ops.is_repo.return_value = True
        git_ops.branch_exists.return_value = False
        git_ops.is_dirty.return_value = False

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            metrics = MetricRegistry()

            setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
            hypothesis = _make_hypothesis(target_metric="val_loss")
            config = _make_config()

            program_path = setup.run(
            hypothesis, config, "val_loss",
            run_command="python train.py --epochs 10",
            baseline={"val_loss": 0.5},
            coding_agent_model="opencode-go/deepseek-v4-pro",
        )

            # Branch was created
            git_ops.create_branch.assert_called_once_with(
                "experiment/abc123def456"
            )
            # Program.md was written
            assert program_path.exists()
            assert program_path.name == "program.md"
            content = program_path.read_text(encoding="utf-8")
            assert "Test hypothesis" in content

    def test_different_metric_builtin_works(self) -> None:
        """Different built-in metric (f1) passes validation."""
        git_ops = MagicMock(spec=GitOperations)
        git_ops.is_repo.return_value = True
        git_ops.branch_exists.return_value = False
        git_ops.is_dirty.return_value = False

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            metrics = MetricRegistry()

            setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)
            hypothesis = _make_hypothesis(target_metric="f1")
            config = _make_config()

            program_path = setup.run(
            hypothesis, config, "f1",
            run_command="python train.py",
            baseline={"f1": 0.8},
            coding_agent_model="test-model",
        )
            assert program_path.exists()
            content = program_path.read_text(encoding="utf-8")
            assert "target_metric: f1" in content
