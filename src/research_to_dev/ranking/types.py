"""Data contracts for the ranking funnel pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from research_to_dev.extraction.types import AcademicContent


@dataclass
class RankedPaper:
    """A paper with its ranking scores and LLM reasoning.

    Produced by the ranking funnel after all 3 stages complete.
    """

    paper: AcademicContent
    """The original academic content object."""

    relevance_score: float
    """LLM-assigned relevance score (0.0–1.0). 0.0 on LLM failure."""

    semantic_score: float
    """Cosine similarity between query and abstract (-1.0 to 1.0). 0.0 on missing abstract or embedder failure."""

    llm_reasoning: str
    """Natural language explanation from the LLM for the relevance score."""

    passed_metadata_filter: bool
    """Whether this paper passed the Stage 1 metadata filter (published_date year >= min_year)."""


@dataclass
class RankingResult:
    """Aggregation of ranking outputs across all 3 stages.

    ``papers`` holds the final ranked list (5–10 entries under normal
    conditions; fewer if input is small or errors occur).
    """

    papers: list[RankedPaper]
    """Final ranked papers after all 3 stages."""

    total_input: int
    """Number of AcademicContent items originally fed to the pipeline."""

    filtered_by_metadata: int
    """Number of papers excluded by the metadata filter (Stage 1)."""

    semantically_ranked: int
    """Number of papers that entered the semantic ranking stage (Stage 2)."""


@dataclass
class RankingError:
    """Metadata about a failure in the ranking pipeline.

    Follows the same pattern as ``ExtractionError`` and ``RetrievalError``.
    Not currently used inside ``RankingResult`` (RF-05 handles degradation
    inline), but available for future error aggregation needs.
    """

    source: str
    """Identifier for what failed (e.g. 'metadata_filter', 'semantic_rank', 'llm_judge')."""

    url: str | None = None
    """Related URL, if applicable."""

    message: str = ""
    """Human-readable error description."""

    exception_type: str | None = None
    """Python exception class name for debugging."""
