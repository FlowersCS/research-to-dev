"""Hypothesis Generator — LLM-powered code improvement hypothesis generation.

Public API
----------
- **Pipeline**: ``HypothesisPipeline`` — orchestrates correlate→generate→reflect→assemble
- **Data**: ``Hypothesis``, ``HypothesisScores``, ``HypothesisResult``, ``HypothesisError``
- **Protocols**: ``HypothesisGenerator`` — pluggable dependency for LLM generation
- **Adapters**: ``OpenAIHypothesisGenerator`` — OpenAI implementation
- **Config**: ``HypothesisConfig`` — constructor-injected configuration
"""

from research_to_dev.hypothesis.adapters import OpenAIHypothesisGenerator
from research_to_dev.hypothesis.pipeline import HypothesisPipeline
from research_to_dev.hypothesis.protocols import HypothesisGenerator
from research_to_dev.hypothesis.types import (
    Hypothesis,
    HypothesisError,
    HypothesisResult,
    HypothesisScores,
)
from research_to_dev.shared.config import HypothesisConfig

__all__ = [
    "Hypothesis",
    "HypothesisConfig",
    "HypothesisError",
    "HypothesisGenerator",
    "HypothesisPipeline",
    "HypothesisResult",
    "HypothesisScores",
    "OpenAIHypothesisGenerator",
]
