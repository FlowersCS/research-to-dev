"""Data contracts for the paper profiling pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from research_to_dev.ranking.types import RankedPaper


@dataclass
class Section:
    """A named section extracted from a paper.

    Produced by the ``SectionExtractor`` — one LLM call extracts all sections
    and their nested claims together (AD-03).
    """

    name: str
    """Section label (e.g. 'abstract', 'methods', 'results', 'conclusion')."""

    content: str
    """Extracted section text."""


@dataclass
class Claim:
    """An assertion-level claim with stable provenance.

    Each claim carries ``paper_id`` and ``section_name`` so downstream
    components (CorrelationMap) can trace it back to its source paper
    and section (AD-06).
    """

    text: str
    """Verifiable assertion extracted from the paper."""

    paper_id: str
    """Stable paper identifier derived from title + source via code (AD-04)."""

    section_name: str
    """Name of the section this claim belongs to."""


@dataclass
class PaperProfile:
    """A wrapper around ``RankedPaper`` enriched with sections and claims.

    Full traceability is preserved (AD-02): ``ranked_paper`` is unmodified,
    and all original ranking scores + ``AcademicContent`` remain accessible
    via ``profile.ranked_paper.paper``.
    """

    ranked_paper: RankedPaper
    """Original ranked paper — all fields preserved."""

    sections: list[Section] = field(default_factory=list)
    """Sections extracted by the LLM extractor."""

    claims: list[Claim] = field(default_factory=list)
    """Assertion-level claims extracted across all sections."""


@dataclass
class ProfilingError:
    """Metadata about a failure during paper profiling.

    Follows the same pattern as ``RankingError`` and ``ExtractionError``.
    """

    paper_title: str
    """Title of the paper that failed."""

    message: str
    """Human-readable error description."""

    exception_type: str | None = None
    """Python exception class name for debugging."""


@dataclass
class ProfilingResult:
    """Aggregation of profiling outputs across all ranked papers.

    ``profiles`` holds one ``PaperProfile`` per input ``RankedPaper``.
    ``errors`` holds failure metadata for papers that couldn't be profiled.
    """

    profiles: list[PaperProfile]
    """Successfully profiled papers."""

    total_input: int
    """Number of ``RankedPaper`` items originally fed to the pipeline."""

    errors: list[ProfilingError] = field(default_factory=list)
    """Failure metadata for papers that errored during profiling."""
