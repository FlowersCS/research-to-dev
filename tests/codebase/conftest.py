"""Pytest configuration and shared fixtures for codebase tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from research_to_dev.shared.config import CodebaseConfig


# -------------------------------------------------------------------
# Config fixtures
# -------------------------------------------------------------------


@pytest.fixture
def codebase_config() -> CodebaseConfig:
    """Default CodebaseConfig with test-friendly parameters."""
    return CodebaseConfig(
        include_private=False,
        max_components_per_module=200,
        model="gpt-4o-mini",
    )


@pytest.fixture
def codebase_config_include_private() -> CodebaseConfig:
    """CodebaseConfig with include_private=True."""
    return CodebaseConfig(
        include_private=True,
        max_components_per_module=200,
        model="gpt-4o-mini",
    )


@pytest.fixture
def codebase_config_capped() -> CodebaseConfig:
    """CodebaseConfig with a low max_components_per_module for cap testing."""
    return CodebaseConfig(
        include_private=False,
        max_components_per_module=5,
        model="gpt-4o-mini",
    )


# -------------------------------------------------------------------
# Protocol mock fixtures
# -------------------------------------------------------------------


@pytest.fixture
def mock_describer() -> AsyncMock:
    """AsyncMock that satisfies the CodebaseDescriber Protocol.

    Returns predictable descriptions and module summaries.
    """
    describer = AsyncMock()
    describer.describe.return_value = {
        "descriptions": [
            {
                "name": "train_model",
                "kind": "function",
                "description": "Trains the model on the given dataset.",
            },
            {
                "name": "evaluate",
                "kind": "function",
                "description": "Evaluates model performance on the test set.",
            },
        ],
        "module_summary": "A machine learning training pipeline module.",
        "key_responsibilities": [
            "Model training orchestration",
            "Performance evaluation",
        ],
    }
    return describer


# -------------------------------------------------------------------
# Temporary Python file fixtures
# -------------------------------------------------------------------


@pytest.fixture
def temp_project_dir() -> str:
    """Create a temporary directory to simulate a Python project.

    Yields the directory path. The directory is cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def write_temp_py_file():
    """Factory fixture — returns a helper to write a .py file in a temp dir.

    Usage::

        def test_something(temp_project_dir, write_temp_py_file):
            path = write_temp_py_file(temp_project_dir, "module.py", "def foo(): pass")
            # path is the full path to the created file
    """

    def _write(root: str, filename: str, content: str) -> str:
        filepath = os.path.join(root, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        Path(filepath).write_text(content)
        return filepath

    return _write
