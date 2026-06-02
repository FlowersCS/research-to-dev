"""Ranking Pipeline — 3-stage sequential funnel for paper relevance.

Orchestrates the three stages (metadata filter → semantic ranking →
LLM judgment) sequentially. Follows the same constructor-injection
pattern as ``ExtractionPipeline``.
"""

from __future__ import annotations

import logging

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.ranking.protocols import Embedder, LLMJudge
from research_to_dev.ranking.stages import llm_judge, metadata_filter, semantic_rank
from research_to_dev.ranking.types import RankedPaper, RankingResult
from research_to_dev.shared.config import RankingConfig


class RankingPipeline:
    """3-stage ranking funnel for academic paper relevance.

    Constructor-injected config and protocol implementations keep the
    pipeline testable without real API calls. Each stage is error-resilient
    (RF-05): a stage failure produces degraded output so downstream stages
    still receive input and the pipeline never raises.
    """

    def __init__(
        self,
        embedder: Embedder,
        judge: LLMJudge,
        *,
        config: RankingConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._embedder = embedder
        self._judge = judge
        self._config = config or RankingConfig()
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rank(
        self, papers: list[AcademicContent], query: str
    ) -> RankingResult:
        """Run the full 3-stage ranking pipeline.

        Args:
            papers: Academic content items to rank.
            query: The research question or topic driving relevance.

        Returns:
            ``RankingResult`` with 5–10 ``RankedPaper`` entries under
            normal conditions. Fewer entries if input is small or LLM
            finds few relevant papers.
        """
        total_input = len(papers)

        # Empty input short-circuit (RF-04 scenario: empty input).
        if total_input == 0:
            return RankingResult(
                papers=[],
                total_input=0,
                filtered_by_metadata=0,
                semantically_ranked=0,
            )

        # ------------------------------------------------------------------
        # Stage 1: Metadata Filter
        # ------------------------------------------------------------------
        filtered, excluded = metadata_filter(
            papers, min_year=self._config.min_year
        )

        # Track which paper *objects* passed the filter — we use identity
        # because AcademicContent is not hashable.
        passed_set = {id(p) for p in filtered}

        # ------------------------------------------------------------------
        # Stage 2: Semantic Ranking
        # ------------------------------------------------------------------
        ranked = await semantic_rank(
            filtered,
            query,
            embedder=self._embedder,
            top_k=self._config.semantic_top_k,
        )
        semantically_ranked = len(ranked)

        # ------------------------------------------------------------------
        # Stage 3: LLM Judgment
        # ------------------------------------------------------------------
        judgments = await llm_judge(
            ranked,
            query,
            judge=self._judge,
            min_papers=self._config.llm_min_papers,
            max_papers=self._config.llm_max_papers,
        )

        # Build RankedPaper objects by mapping paper_index back.
        paper_lookup = {i: (paper, score) for i, (paper, score) in enumerate(ranked)}
        final_papers: list[RankedPaper] = []
        for j in judgments:
            idx = j.get("paper_index")
            if idx is not None and idx in paper_lookup:
                paper, semantic_score = paper_lookup[idx]
                final_papers.append(
                    RankedPaper(
                        paper=paper,
                        relevance_score=j.get("relevance_score", 0.0),
                        semantic_score=semantic_score,
                        llm_reasoning=j.get("reasoning", ""),
                        passed_metadata_filter=id(paper) in passed_set,
                    )
                )

        return RankingResult(
            papers=final_papers,
            total_input=total_input,
            filtered_by_metadata=excluded,
            semantically_ranked=semantically_ranked,
        )
