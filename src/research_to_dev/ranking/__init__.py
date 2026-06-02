"""Ranking Funnel — 3-stage pipeline for research paper relevance.

Public API
----------
- **Pipeline**: ``RankingPipeline`` — orchestrates the 3-stage funnel
- **Data**: ``RankedPaper``, ``RankingResult``, ``RankingError``
- **Protocols**: ``Embedder``, ``LLMJudge`` — pluggable dependencies
- **Adapters**: ``OpenAIEmbedder``, ``OpenAIJudge`` — OpenAI implementations
- **Config**: ``RankingConfig`` — constructor-injected configuration
"""

from research_to_dev.ranking.adapters import OpenAIEmbedder, OpenAIJudge
from research_to_dev.ranking.pipeline import RankingPipeline
from research_to_dev.ranking.protocols import Embedder, LLMJudge
from research_to_dev.ranking.types import RankedPaper, RankingError, RankingResult
from research_to_dev.shared.config import RankingConfig

__all__ = [
    "Embedder",
    "LLMJudge",
    "OpenAIEmbedder",
    "OpenAIJudge",
    "RankedPaper",
    "RankingConfig",
    "RankingError",
    "RankingPipeline",
    "RankingResult",
]
