"""Three-stage ranking funnel stage implementations.

Each stage is a pure async function with no class coupling.
The ``RankingPipeline`` orchestrates them sequentially.
"""

from __future__ import annotations

import logging

import numpy as np

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.ranking.protocols import Embedder, LLMJudge

_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Stage 1: Metadata Filter
# ------------------------------------------------------------------


def metadata_filter(
    papers: list[AcademicContent],
    min_year: int,
) -> tuple[list[AcademicContent], int]:
    """Exclude papers whose ``published_date`` year is before ``min_year``.

    Papers with missing ``published_date`` pass through unfiltered.
    Unparseable date strings also pass through (best effort).

    Args:
        papers: Input papers to filter.
        min_year: Minimum publication year (inclusive).

    Returns:
        Tuple of ``(filtered_papers, excluded_count)``.
    """
    filtered: list[AcademicContent] = []
    excluded = 0

    for paper in papers:
        if paper.published_date is None:
            filtered.append(paper)
            continue

        try:
            year = int(paper.published_date[:4])
        except (ValueError, IndexError):
            # Unparseable date — let it pass
            filtered.append(paper)
            continue

        if year >= min_year:
            filtered.append(paper)
        else:
            excluded += 1

    return filtered, excluded


# ------------------------------------------------------------------
# Stage 2: Semantic Ranking
# ------------------------------------------------------------------


async def semantic_rank(
    papers: list[AcademicContent],
    query: str,
    *,
    embedder: Embedder,
    top_k: int,
) -> list[tuple[AcademicContent, float]]:
    """Embed query and paper abstracts, then rank by cosine similarity.

    Papers with missing or empty ``abstract`` receive ``semantic_score=0.0``
    but are still eligible for downstream LLM judgment (title-only).

    On embedder failure (RF-05): all papers receive ``semantic_score=0.0``
    and the pipeline continues.

    Args:
        papers: Papers to rank (already metadata-filtered).
        query: The research question or topic.
        embedder: Protocol implementation for text embedding.
        top_k: Number of top-scoring papers to select.

    Returns:
        List of ``(paper, semantic_score)`` tuples, sorted descending
        by score, capped at ``top_k`` entries.
    """
    try:
        abstracts = [p.abstract or "" for p in papers]

        query_vec = np.array(await embedder.embed_query(query))
        paper_vecs = np.array(await embedder.embed(abstracts))

        norm_q = np.linalg.norm(query_vec)

        scored: list[tuple[AcademicContent, float]] = []
        for i, paper in enumerate(papers):
            if not paper.abstract:
                scored.append((paper, 0.0))
                continue

            norm_a = np.linalg.norm(paper_vecs[i])
            if norm_q == 0.0 or norm_a == 0.0:
                scored.append((paper, 0.0))
                continue

            sim = float(
                np.dot(query_vec, paper_vecs[i]) / (norm_q * norm_a)
            )
            scored.append((paper, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    except Exception as exc:
        _logger.warning("Semantic ranking failed: %s", exc)
        # RF-05: degraded output — all papers score 0.0
        result = [(p, 0.0) for p in papers]
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_k]


# ------------------------------------------------------------------
# Stage 3: LLM Judgment
# ------------------------------------------------------------------


async def llm_judge(
    papers: list[tuple[AcademicContent, float]],
    query: str,
    *,
    judge: LLMJudge,
    min_papers: int,
    max_papers: int,
) -> list[dict]:
    """Judge paper relevance via a single batched LLM prompt.

    Sends all top-K papers in one call (AD-09). The LLM returns
    structured JSON with ``paper_index``, ``relevance_score``, and
    ``reasoning``.

    On LLM failure (RF-05): all papers receive ``relevance_score=0.0``
    with empty ``reasoning``, and a valid result is still returned.

    Args:
        papers: Top-K ranked papers with their semantic scores.
        query: The research question or topic.
        judge: Protocol implementation for LLM judgment.
        min_papers: Minimum papers to return (informational, LLM may
            return fewer if truly nothing is relevant).
        max_papers: Maximum papers to return.

    Returns:
        List of dicts with keys ``paper_index``, ``relevance_score``,
        ``reasoning``. Sorted by ``relevance_score`` descending, capped
        at ``max_papers``.
    """
    paper_dicts = [
        {
            "paper_index": i,
            "title": paper.title,
            "abstract": paper.abstract or "",
            "semantic_score": score,
        }
        for i, (paper, score) in enumerate(papers)
    ]

    try:
        results = await judge.judge(query, paper_dicts)

        # Keep only results with well-formed relevance scores
        valid = [
            r
            for r in results
            if isinstance(r.get("relevance_score"), (int, float))
            and 0.0 <= r["relevance_score"] <= 1.0
        ]

        valid.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)
        return valid[:max_papers]

    except Exception as exc:
        _logger.warning("LLM judgment failed: %s", exc)
        # RF-05: degraded output — all papers score 0.0, empty reasoning
        return [
            {
                "paper_index": i,
                "relevance_score": 0.0,
                "reasoning": "",
            }
            for i in range(len(papers))
        ]
