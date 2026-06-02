"""Shared configuration dataclasses for the research-to-dev pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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
