"""Shared configuration dataclasses for the research-to-dev pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProfileConfig:
    """Configuration for the Paper Profiling pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    model: str = "gpt-4o-mini"
    """OpenAI chat model used for section and claim extraction."""

    max_claims_per_paper: int = 20
    """Maximum number of claims to keep per paper (caps output size)."""


@dataclass
class RankingConfig:
    """Configuration for the Ranking Funnel pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    min_year: int = field(default_factory=lambda: datetime.now().year - 5)
    """Hard filter threshold — papers dated before this year are excluded."""

    semantic_top_k: int = 20
    """Number of top papers to select after semantic ranking (Stage 2)."""

    llm_min_papers: int = 5
    """Minimum number of papers the LLM should return (Stage 3)."""

    llm_max_papers: int = 10
    """Maximum number of papers the LLM should return (Stage 3)."""

    embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model used for semantic ranking."""

    llm_model: str = "gpt-4o-mini"
    """OpenAI chat model used for relevance judgment."""


@dataclass
class CorrelationMapConfig:
    """Configuration for the CorrelationMap pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    similarity_threshold: float = 0.7
    """Cosine similarity cutoff — pairs below this score are discarded (CM-CFG-01)."""

    embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model used for similarity computation (CM-CFG-01)."""

    llm_model: str = "gpt-4o-mini"
    """OpenAI chat model used for correlation classification (CM-CFG-01)."""


@dataclass
class CodebaseConfig:
    """Configuration for the Codebase Analysis pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    include_private: bool = False
    """Whether to include ``_``-prefixed functions and classes."""

    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "test_*.py",
            "*_test.py",
            "setup.py",
            "conftest.py",
        ]
    )
    """Glob patterns for files to exclude from scanning."""

    include_patterns: list[str] = field(default_factory=lambda: ["*.py"])
    """Glob patterns for files to include in scanning."""

    max_components_per_module: int = 200
    """Maximum number of components to extract from a single module."""

    model: str = "gpt-4o-mini"
    """OpenAI chat model used for component description and module summaries."""
