"""Correlation Map — embedding-based cross-reference between paper claims and code.

Public API
----------
- **Pipeline**: ``CorrelationPipeline`` — orchestrates embed→cosine→filter→classify→assemble
- **Data**: ``Correlation``, ``ModuleCorrelation``, ``CorrelationMap``, ``CorrelationError``
- **Protocols**: ``CorrelationClassifier`` — pluggable dependency for LLM classification
- **Adapters**: ``OpenAICorrelationClassifier`` — OpenAI implementation
- **Config**: ``CorrelationMapConfig`` — constructor-injected configuration
"""

from research_to_dev.correlation.adapters import OpenAICorrelationClassifier
from research_to_dev.correlation.pipeline import CorrelationPipeline
from research_to_dev.correlation.protocols import CorrelationClassifier
from research_to_dev.correlation.types import (
    Correlation,
    CorrelationError,
    CorrelationMap,
    ModuleCorrelation,
)
from research_to_dev.shared.config import CorrelationMapConfig

__all__ = [
    "Correlation",
    "CorrelationClassifier",
    "CorrelationError",
    "CorrelationMap",
    "CorrelationMapConfig",
    "CorrelationPipeline",
    "ModuleCorrelation",
    "OpenAICorrelationClassifier",
]
