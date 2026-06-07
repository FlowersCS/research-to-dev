"""Tests for ProgramWriter — YAML frontmatter + Markdown body generation.

Verifies ES-05A through ES-05D: frontmatter contains canonical fields,
Markdown body has Objective and Agent Instructions sections, success
criteria appear in both locations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from research_to_dev.experiment.program import ProgramWriter
from research_to_dev.hypothesis.types import (
    Hypothesis,
    HypothesisScores,
)
from research_to_dev.shared.config import ExperimentConfig


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


class TestProgramWriterStructure:
    """ES-05A: program.md starts with YAML frontmatter and has two sections."""

    def test_program_starts_with_yaml_frontmatter(self) -> None:
        """File starts with YAML frontmatter delimited by ---."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = writer.write(hypothesis, config, "val_loss")

            content = path.read_text(encoding="utf-8")
            assert content.startswith("---")

    def test_frontmatter_contains_canonical_fields(self) -> None:
        """YAML frontmatter has all required canonical fields (ES-05A)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = writer.write(hypothesis, config, "val_loss")

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

    def test_body_contains_objective_section(self) -> None:
        """Body has '## Experiment Objective' heading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = writer.write(hypothesis, config, "val_loss")

            content = path.read_text(encoding="utf-8")
            assert "## Experiment Objective" in content

    def test_body_contains_agent_instructions_section(self) -> None:
        """Body has '## Agent Instructions' heading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config()
            path = writer.write(hypothesis, config, "val_loss")

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
            path = writer.write(hypothesis, config, "val_loss")

            content = path.read_text(encoding="utf-8")
            assert "My specific test title" in content

    def test_objective_includes_approach(self) -> None:
        """Objective contains the approach."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(approach="Insert nn.Dropout(0.3)")
            config = _make_config()
            path = writer.write(hypothesis, config, "val_loss")

            content = path.read_text(encoding="utf-8")
            assert "Insert nn.Dropout(0.3)" in content

    def test_objective_includes_expected_improvement(self) -> None:
        """Objective contains expected improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(expected_improvement="Reduce loss by 15%")
            config = _make_config()
            path = writer.write(hypothesis, config, "val_loss")

            content = path.read_text(encoding="utf-8")
            assert "Reduce loss by 15%" in content

    def test_objective_includes_supporting_papers(self) -> None:
        """Objective lists supporting papers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(supporting_papers=["Paper A", "Paper B"])
            config = _make_config()
            path = writer.write(hypothesis, config, "val_loss")

            content = path.read_text(encoding="utf-8")
            assert "Paper A" in content
            assert "Paper B" in content

    def test_objective_includes_code_changes(self) -> None:
        """Objective includes code changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis(code_changes="src/model.py")
            config = _make_config()
            path = writer.write(hypothesis, config, "val_loss")

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
            path = writer.write(hypothesis, config, "val_loss")

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
            path = writer.write(hypothesis, config, "val_loss")

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
            path = writer.write(hypothesis, config, "val_loss")

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
            path = writer.write(hypothesis, config, "f1")

            content = path.read_text(encoding="utf-8")
            assert "`f1`" in content

    def test_agent_instructions_include_time_budget(self) -> None:
        """Agent instructions include time_budget from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config(time_budget="2h")
            path = writer.write(hypothesis, config, "val_loss")

            content = path.read_text(encoding="utf-8")
            assert "2h" in content

    def test_agent_instructions_include_max_iterations(self) -> None:
        """Agent instructions include max_iterations from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ProgramWriter(base_dir=tmpdir)
            hypothesis = _make_hypothesis()
            config = _make_config(max_iterations=5)
            path = writer.write(hypothesis, config, "val_loss")

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
            path = writer.write(hypothesis, config, "val_loss")

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
            path = writer.write(hypothesis, config, "val_loss")

            assert isinstance(path, Path)
            assert path.name == "program.md"
