"""Unit tests for codebase data types and configuration."""

from __future__ import annotations

from research_to_dev.codebase.types import (
    CodebaseContext,
    CodebaseError,
    Component,
    ModuleSummary,
)
from research_to_dev.shared.config import CodebaseConfig


# -------------------------------------------------------------------
# CodebaseConfig (CA-13)
# -------------------------------------------------------------------


class TestCodebaseConfig:
    """Verify CodebaseConfig dataclass creation and defaults (CA-13)."""

    def test_default_values_are_correct(self) -> None:
        """CA-13: include_private=False, max_components_per_module=200."""
        cfg = CodebaseConfig()
        assert cfg.include_private is False
        assert cfg.max_components_per_module == 200

    def test_model_default_is_set(self) -> None:
        """model defaults to 'gpt-4o-mini'."""
        cfg = CodebaseConfig()
        assert cfg.model == "gpt-4o-mini"

    def test_exclude_patterns_default(self) -> None:
        """exclude_patterns defaults include test files and setup.py."""
        cfg = CodebaseConfig()
        assert "test_*.py" in cfg.exclude_patterns
        assert "*_test.py" in cfg.exclude_patterns
        assert "setup.py" in cfg.exclude_patterns
        assert "conftest.py" in cfg.exclude_patterns

    def test_include_patterns_default(self) -> None:
        """include_patterns defaults to ['*.py']."""
        cfg = CodebaseConfig()
        assert cfg.include_patterns == ["*.py"]

    def test_custom_values_preserved_exactly(self) -> None:
        """Custom values preserve exactly."""
        cfg = CodebaseConfig(
            include_private=True,
            max_components_per_module=50,
            model="gpt-4o",
            exclude_patterns=["*.txt"],
            include_patterns=["*.custom"],
        )
        assert cfg.include_private is True
        assert cfg.max_components_per_module == 50
        assert cfg.model == "gpt-4o"
        assert cfg.exclude_patterns == ["*.txt"]
        assert cfg.include_patterns == ["*.custom"]


# -------------------------------------------------------------------
# Component
# -------------------------------------------------------------------


class TestComponent:
    """Verify Component dataclass creation."""

    def test_full_fields(self) -> None:
        comp = Component(
            name="train_model",
            kind="function",
            file_path="/src/training/pipeline.py",
            line_start=10,
            line_end=25,
            signature="def train_model(data: Dataset, epochs: int = 10) -> Model",
            description="Trains the model on the given dataset.",
            module_name="training.pipeline",
            docstring="Train the model using the configured optimizer.",
        )
        assert comp.name == "train_model"
        assert comp.kind == "function"
        assert comp.file_path == "/src/training/pipeline.py"
        assert comp.line_start == 10
        assert comp.line_end == 25
        assert comp.signature == "def train_model(data: Dataset, epochs: int = 10) -> Model"
        assert comp.description == "Trains the model on the given dataset."
        assert comp.module_name == "training.pipeline"
        assert comp.docstring == "Train the model using the configured optimizer."

    def test_defaults(self) -> None:
        """docstring defaults to None."""
        comp = Component(
            name="helper",
            kind="function",
            file_path="/src/utils.py",
            line_start=1,
            line_end=3,
            signature="def helper() -> None",
            description="",
            module_name="utils",
        )
        assert comp.docstring is None
        assert comp.description == ""

    def test_class_component(self) -> None:
        """kind='class' with method-like signature."""
        comp = Component(
            name="ModelTrainer",
            kind="class",
            file_path="/src/trainer.py",
            line_start=5,
            line_end=50,
            signature="class ModelTrainer(ABC)",
            description="Base class for model trainers.",
            module_name="trainer",
        )
        assert comp.kind == "class"
        assert comp.signature == "class ModelTrainer(ABC)"

    def test_method_component(self) -> None:
        """kind='method' for class-level methods."""
        comp = Component(
            name="fit",
            kind="method",
            file_path="/src/trainer.py",
            line_start=15,
            line_end=30,
            signature="def fit(self, X: np.ndarray, y: np.ndarray) -> None",
            description="Fits the model to the training data.",
            module_name="trainer",
        )
        assert comp.kind == "method"


# -------------------------------------------------------------------
# ModuleSummary
# -------------------------------------------------------------------


class TestModuleSummary:
    """Verify ModuleSummary dataclass creation."""

    def test_full_fields(self) -> None:
        summary = ModuleSummary(
            module_name="training.pipeline",
            file_path="/src/training/pipeline.py",
            summary="Orchestrates the end-to-end training process.",
            key_responsibilities=[
                "Data loading and preprocessing",
                "Model training loop",
                "Checkpoint management",
            ],
        )
        assert summary.module_name == "training.pipeline"
        assert summary.file_path == "/src/training/pipeline.py"
        assert summary.summary == "Orchestrates the end-to-end training process."
        assert len(summary.key_responsibilities) == 3
        assert summary.key_responsibilities[0] == "Data loading and preprocessing"

    def test_default_empty_list(self) -> None:
        """key_responsibilities defaults to empty list."""
        summary = ModuleSummary(
            module_name="empty",
            file_path="/src/empty.py",
            summary="Nothing here.",
        )
        assert summary.key_responsibilities == []


# -------------------------------------------------------------------
# CodebaseContext
# -------------------------------------------------------------------


class TestCodebaseContext:
    """Verify CodebaseContext dataclass creation."""

    def test_full_fields(self) -> None:
        comp = Component(
            name="foo",
            kind="function",
            file_path="/src/a.py",
            line_start=1,
            line_end=2,
            signature="def foo() -> None",
            description="Does foo.",
            module_name="a",
        )
        summary = ModuleSummary(
            module_name="a",
            file_path="/src/a.py",
            summary="Module A.",
        )

        ctx = CodebaseContext(
            project_name="my-project",
            project_path="/home/user/my-project",
            language="python",
            components=[comp],
            modules=[summary],
            total_files_scanned=1,
            total_components_found=1,
        )
        assert ctx.project_name == "my-project"
        assert ctx.project_path == "/home/user/my-project"
        assert ctx.language == "python"
        assert len(ctx.components) == 1
        assert len(ctx.modules) == 1
        assert ctx.total_files_scanned == 1
        assert ctx.total_components_found == 1

    def test_defaults(self) -> None:
        """components, modules default to empty lists; stats default to 0."""
        ctx = CodebaseContext(
            project_name="empty",
            project_path="/tmp/empty",
            language="python",
        )
        assert ctx.components == []
        assert ctx.modules == []
        assert ctx.total_files_scanned == 0
        assert ctx.total_components_found == 0


# -------------------------------------------------------------------
# CodebaseError
# -------------------------------------------------------------------


class TestCodebaseError:
    """Verify CodebaseError dataclass creation and defaults."""

    def test_full_fields(self) -> None:
        error = CodebaseError(
            file_path="/src/broken.py",
            message="Syntax error in module",
            exception_type="SyntaxError",
        )
        assert error.file_path == "/src/broken.py"
        assert error.message == "Syntax error in module"
        assert error.exception_type == "SyntaxError"

    def test_minimal_fields(self) -> None:
        error = CodebaseError(
            file_path="/src/timeout.py",
            message="Scan timed out",
        )
        assert error.file_path == "/src/timeout.py"
        assert error.message == "Scan timed out"
        assert error.exception_type is None
