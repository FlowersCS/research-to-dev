"""Tests for ProgramWriter — YAML frontmatter + Markdown body generation.

Verifies ES-05A through ES-05D: frontmatter contains canonical fields,
Markdown body has Objective and Agent Instructions sections, success
criteria appear in both locations.

Phase 7 (AE-31): tests for new frontmatter fields (run_command, baseline,
direction, coding_agent_model) and roundtrip write→parse verification.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from research_to_dev.experiment.program import ProgramWriter
from research_to_dev.experiment.program_reader import (
    ProgramSpec,
    parse_program_md,
)
from research_to_dev.hypothesis.types import (
    Hypothesis,
    HypothesisScores,
)
from research_to_dev.shared.config import ExperimentConfig


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _make_hypothesis(**overrides: object) -> Hypothesis:
    """Build a test Hypothesis with sensible defaults."""
    defaults: dict[str, object] = {
        "id": "abc123def456",
        "title": "Add dropout to reduce overfitting",
        "description": (
            "Insert nn.Dropout layers to regularize the model and reduce "
            "overfitting on validation set."
        ),
        "approach": "Add nn.Dropout(0.3) after each linear layer in model.py",
        "supporting_papers": ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"],
        "target_metric": "val_loss",
        "expected_improvement": (
            "Reduce val_loss by 15% while maintaining training convergence"
        ),
        "code_changes": "src/model.py, src/train.py",
        "scores": HypothesisScores(relevance=8, feasibility=9, evidence=7),
        "composite": 8.0,
        "success_criteria": "val_loss < 0.5 after 10 epochs",
    }
    defaults.update(overrides)
    return Hypothesis(**defaults)  # type: ignore[arg-type]


def _make_config(**overrides: object) -> ExperimentConfig:
    """Build a test ExperimentConfig."""
    defaults: dict[str, object] = {
        "time_budget": "30m",
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)  # type: ignore[arg-type]


def _write(
    writer: ProgramWriter,
    hypothesis: Hypothesis,
    config: ExperimentConfig,
    target_metric: str = "val_loss",
    *,
    run_command: str = "python train.py",
    baseline: dict[str, float] | None = None,
    coding_agent_model: str = "test-model",
    direction: str | None = None,
) -> Path:
    """Helper wrapper — provides sensible defaults for all new fields."""
    if baseline is None:
        baseline = {"val_loss": 0.5}
    return writer.write(
        hypothesis,
        config,
        target_metric,
        run_command=run_command,
        baseline=baseline,
        coding_agent_model=coding_agent_model,
        direction=direction,
    )


# ------------------------------------------------------------------
# Existing tests (updated to use _write helper)
# ------------------------------------------------------------------


class TestProgramWriterStructure:
    """ES-05A: program.md starts with YAML frontmatter and has two sections."""

    def test_program_starts_with_yaml_frontmatter(self) -> None:
        """File starts with YAML frontmatter delimited by ---."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert content.startswith("---")

    def test_frontmatter_contains_canonical_fields(self) -> None:
        """YAML frontmatter has all required canonical fields (ES-05A)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            # Extract frontmatter
            parts = content.split("---", 2)
            assert len(parts) >= 3
            frontmatter = yaml.safe_load(parts[1])

            assert frontmatter["hypothesis_id"] == "abc123def456"
            assert frontmatter["target_metric"] == "val_loss"
            assert frontmatter["success_criteria"] == "val_loss < 0.5 after 10 epochs"
            assert frontmatter["time_budget"] == "30m"
            assert frontmatter["max_iterations"] == 10
            # New fields (AE-28)
            assert frontmatter["run_command"] == "python train.py"
            assert frontmatter["baseline"] == {"val_loss": 0.5}
            assert frontmatter["coding_agent_model"] == "test-model"

    def test_body_contains_objective_section(self) -> None:
        """Body has '## Experiment Objective' heading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "## Experiment Objective" in content

    def test_body_contains_agent_instructions_section(self) -> None:
        """Body has '## Agent Instructions' heading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "## Agent Instructions" in content


class TestProgramWriterContent:
    """ES-05B: Objective section contains hypothesis fields."""

    def test_objective_includes_title(self) -> None:
        """Objective contains the hypothesis title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(title="My specific test title")
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "My specific test title" in content

    def test_objective_includes_approach(self) -> None:
        """Objective contains the approach."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(approach="Insert nn.Dropout(0.3)")
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "Insert nn.Dropout(0.3)" in content

    def test_objective_includes_expected_improvement(self) -> None:
        """Objective contains expected improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(expected_improvement="Reduce loss by 15%")
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "Reduce loss by 15%" in content

    def test_objective_includes_supporting_papers(self) -> None:
        """Objective lists supporting papers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(supporting_papers=["Paper A", "Paper B"])
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "Paper A" in content
            assert "Paper B" in content

    def test_objective_includes_code_changes(self) -> None:
        """Objective includes code changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(code_changes="src/model.py")
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "src/model.py" in content


class TestSuccessCriteriaInBothLocations:
    """ES-05D: success_criteria in frontmatter AND objective prose."""

    def test_success_criteria_in_frontmatter(self) -> None:
        """success_criteria appears in YAML frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(
                success_criteria="p < 0.01 with N=100"
            )
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])
            assert frontmatter["success_criteria"] == "p < 0.01 with N=100"

    def test_success_criteria_in_objective_prose(self) -> None:
        """success_criteria appears in objective prose."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(
                success_criteria="p < 0.01 with N=100"
            )
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "p < 0.01 with N=100" in content
            assert "**Success Criteria**" in content


class TestAgentInstructionsTemplate:
    """ES-05C: Agent instructions are a fixed template, not LLM-generated."""

    def test_agent_instructions_contain_fixed_text(self) -> None:
        """Agent instructions contain the fixed template text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "You are testing a hypothesis about code improvement" in content
            assert "Modify the codebase" in content
            assert "Stop when success criteria are met" in content

    def test_agent_instructions_include_target_metric(self) -> None:
        """Agent instructions include the target_metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = _write(writer, hypothesis, config, target_metric="f1")

            content = path.read_text(encoding="utf-8")
            assert "`f1`" in content

    def test_agent_instructions_include_time_budget(self) -> None:
        """Agent instructions include time_budget from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config(time_budget="2h")
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "2h" in content

    def test_agent_instructions_include_max_iterations(self) -> None:
        """Agent instructions include max_iterations from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config(max_iterations=5)
            path = _write(writer, hypothesis, config)

            content = path.read_text(encoding="utf-8")
            assert "5" in content


class TestProgramWriterPath:
    """ProgramWriter creates the correct directory structure."""

    def test_writer_creates_experiment_directory(self) -> None:
        """Writer creates .research-to-dev/experiments/<id>/program.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(id="testid")
            config = _make_config()
            path = _write(writer, hypothesis, config)

            expected_dir = Path(tmpdir) / "testid"
            assert expected_dir.exists()
            assert path == expected_dir / "program.md"
            assert path.exists()

    def test_writer_returns_path(self) -> None:
        """write() returns the Path to the created file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = _write(writer, hypothesis, config)

            assert isinstance(path, Path)
            assert path.name == "program.md"


# ======================================================================
# Phase 7 (AE-31): New frontmatter fields and roundtrip tests
# ======================================================================


class TestProgramWriterNewFrontmatter:
    """AE-28, AE-31: ProgramWriter emits run_command, baseline, direction, coding_agent_model."""

    def test_run_command_in_frontmatter(self) -> None:
        """run_command is emitted in YAML frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            path = _write(
                writer, _make_hypothesis(), _make_config(),
                run_command="python train.py --epochs 20",
            )
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])
            assert frontmatter["run_command"] == "python train.py --epochs 20"

    def test_baseline_as_yaml_dict(self) -> None:
        """baseline is emitted as a YAML dict mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            path = _write(
                writer, _make_hypothesis(), _make_config(),
                baseline={"val_accuracy": 0.72, "val_loss": 0.45},
            )
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])
            assert isinstance(frontmatter["baseline"], dict)
            assert frontmatter["baseline"]["val_accuracy"] == 0.72
            assert frontmatter["baseline"]["val_loss"] == 0.45

    def test_coding_agent_model_in_frontmatter(self) -> None:
        """coding_agent_model is emitted in YAML frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            path = _write(
                writer, _make_hypothesis(), _make_config(),
                coding_agent_model="opencode-go/deepseek-v4-pro",
            )
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])
            assert frontmatter["coding_agent_model"] == "opencode-go/deepseek-v4-pro"

    def test_direction_emitted_when_explicit(self) -> None:
        """direction is emitted when explicitly provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            path = _write(
                writer, _make_hypothesis(), _make_config(),
                direction="maximize",
            )
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])
            assert frontmatter["direction"] == "maximize"

    def test_direction_omitted_when_not_provided(self) -> None:
        """direction is NOT in frontmatter when omitted (program_reader infers it)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            path = _write(
                writer, _make_hypothesis(), _make_config(),
                direction=None,
            )
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])
            assert "direction" not in frontmatter

    def test_new_fields_after_existing_ones(self) -> None:
        """New fields appear in the frontmatter YAML, maintaining order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            path = _write(writer, _make_hypothesis(), _make_config())
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            frontmatter = yaml.safe_load(parts[1])

            # All 8 mandatory fields present
            assert "hypothesis_id" in frontmatter
            assert "target_metric" in frontmatter
            assert "success_criteria" in frontmatter
            assert "time_budget" in frontmatter
            assert "max_iterations" in frontmatter
            assert "run_command" in frontmatter
            assert "baseline" in frontmatter
            assert "coding_agent_model" in frontmatter


class TestProgramWriterValidation:
    """AE-28: ProgramWriter fails loud for missing mandatory fields."""

    def test_write_raises_on_empty_run_command(self) -> None:
        """Empty run_command raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            with pytest.raises(ValueError, match="run_command must not be empty"):
                writer.write(
                    _make_hypothesis(), _make_config(), "val_loss",
                    run_command="   ",
                    baseline={"val_loss": 0.5},
                    coding_agent_model="test-model",
                )

    def test_write_raises_on_empty_baseline(self) -> None:
        """Empty baseline dict raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            with pytest.raises(ValueError, match="baseline must not be empty"):
                writer.write(
                    _make_hypothesis(), _make_config(), "val_loss",
                    run_command="python train.py",
                    baseline={},
                    coding_agent_model="test-model",
                )

    def test_write_raises_on_empty_coding_agent_model(self) -> None:
        """Empty coding_agent_model raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            with pytest.raises(ValueError, match="coding_agent_model must not be empty"):
                writer.write(
                    _make_hypothesis(), _make_config(), "val_loss",
                    run_command="python train.py",
                    baseline={"val_loss": 0.5},
                    coding_agent_model="",
                )


class TestProgramWriterRoundtrip:
    """AE-31: Write with ProgramWriter → parse with ProgramReader → verify fields."""

    def test_roundtrip_all_fields(self) -> None:
        """All frontmatter fields survive write→parse roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(
                success_criteria="val_accuracy >= 0.90",
            )
            config = _make_config(time_budget="1h", max_iterations=5)
            path = _write(
                writer, hypothesis, config,
                target_metric="val_accuracy",
                run_command="python evaluate.py",
                baseline={"val_accuracy": 0.72},
                coding_agent_model="opencode-go/deepseek-v4-pro",
                direction="maximize",
            )

            spec = parse_program_md(path)

            assert isinstance(spec, ProgramSpec)
            assert spec.hypothesis_id == "abc123def456"
            assert spec.target_metric == "val_accuracy"
            assert spec.success_criteria == "val_accuracy >= 0.90"
            assert spec.time_budget == "1h"
            assert spec.max_iterations == 5
            assert spec.run_command == "python evaluate.py"
            assert spec.baseline == {"val_accuracy": 0.72}
            assert spec.coding_agent_model == "opencode-go/deepseek-v4-pro"
            assert spec.direction == "maximize"
            assert spec.time_budget_seconds == 3600
            assert "Experiment Objective" in spec.body

    def test_roundtrip_direction_inferred_when_omitted(self) -> None:
        """Direction is inferred from success_criteria when not in frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(
                success_criteria="val_loss <= 0.3",
            )
            config = _make_config()
            path = _write(
                writer, hypothesis, config,
                target_metric="val_loss",
                direction=None,  # omit direction
            )

            spec = parse_program_md(path)
            assert spec.direction == "minimize"  # inferred from "<="

    def test_roundtrip_single_baseline_entry(self) -> None:
        """Roundtrip with a single baseline key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            path = _write(
                writer, _make_hypothesis(), _make_config(),
                baseline={"accuracy": 0.85},
            )

            spec = parse_program_md(path)
            assert spec.baseline == {"accuracy": 0.85}

    def test_roundtrip_multi_baseline_entry(self) -> None:
        """Roundtrip with multiple baseline keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            path = _write(
                writer, _make_hypothesis(), _make_config(),
                baseline={"accuracy": 0.85, "f1": 0.72, "bleu": 25.3},
            )

            spec = parse_program_md(path)
            assert spec.baseline == {"accuracy": 0.85, "f1": 0.72, "bleu": 25.3}
