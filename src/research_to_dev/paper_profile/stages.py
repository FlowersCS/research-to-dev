"""Pure stage functions for the paper profiling pipeline.

Each stage is a pure async function with no class coupling.
The ``ProfilingPipeline`` orchestrates them sequentially.
"""

from __future__ import annotations

import hashlib
import logging

from research_to_dev.paper_profile.protocols import SectionExtractor
from research_to_dev.paper_profile.types import (
    Claim,
    PaperProfile,
    ProfilingError,
    Section,
)
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import ProfileConfig

_logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paper ID Construction (AD-04)
# ------------------------------------------------------------------


def build_paper_id(url: str | None, source: str, title: str) -> str:
    """Construct a stable paper identifier from title and source.

    Prefers ``url`` when available (most stable). Falls back to a
    deterministic hash of ``source:title`` (AD-04).

    Args:
        url: Paper URL (preferred).
        source: Paper source (e.g. 'arxiv', 'semantic_scholar').
        title: Paper title.

    Returns:
        A stable string identifier for the paper.
    """
    if url:
        return url
    digest = hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]
    return f"{source}:{digest}"


# ------------------------------------------------------------------
# Paper Profiling Stage
# ------------------------------------------------------------------


async def profile_paper(
    ranked_paper: RankedPaper,
    *,
    extractor: SectionExtractor,
    config: ProfileConfig | None = None,
) -> tuple[PaperProfile, ProfilingError | None]:
    """Extract sections and claims from a single ranked paper.

    Calls the ``SectionExtractor`` once (AD-03), validates the JSON
    response, constructs ``Section`` and ``Claim`` domain objects,
    and wraps them in a ``PaperProfile`` around the original
    ``RankedPaper`` (AD-02).

    On ANY failure (RF-05): returns a degraded ``PaperProfile`` with
    empty sections and claims, plus a ``ProfilingError`` describing
    the failure.

    Args:
        ranked_paper: The ranked paper to profile.
        extractor: Protocol implementation for section extraction.
        config: Optional profile configuration.

    Returns:
        Tuple of ``(PaperProfile, ProfilingError | None)``.
    """
    if config is None:
        config = ProfileConfig()

    paper = ranked_paper.paper
    paper_id = build_paper_id(paper.url, paper.source, paper.title)
    abstract = paper.abstract or ""

    try:
        raw = await extractor.extract(paper.title, abstract)
    except Exception as exc:
        _logger.warning("Extractor failed for paper '%s': %s", paper.title, exc)
        profile = PaperProfile(ranked_paper=ranked_paper)
        error = ProfilingError(
            paper_title=paper.title,
            message=str(exc),
            exception_type=type(exc).__name__,
        )
        return profile, error

    # Parse and validate the response
    try:
        sections_data = raw.get("sections")
        if not isinstance(sections_data, list):
            raise ValueError(
                f"Expected 'sections' list, got {type(sections_data).__name__}"
            )
    except Exception as exc:
        _logger.warning("Invalid LLM response for paper '%s': %s", paper.title, exc)
        profile = PaperProfile(ranked_paper=ranked_paper)
        error = ProfilingError(
            paper_title=paper.title,
            message=f"Invalid response structure: {exc}",
            exception_type=type(exc).__name__,
        )
        return profile, error

    # Construct Section and Claim objects
    sections: list[Section] = []
    claims: list[Claim] = []
    seen_claims: set[str] = set()

    for s in sections_data:
        if not isinstance(s, dict):
            continue

        name = s.get("name")
        content = s.get("content")
        if not isinstance(name, str) or not isinstance(content, str):
            continue

        sections.append(Section(name=name, content=content))

        raw_claims = s.get("claims")
        if isinstance(raw_claims, list):
            for c in raw_claims:
                if not isinstance(c, dict):
                    continue
                text = c.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                if text in seen_claims:
                    _logger.warning(
                        "Duplicate claim text in paper '%s', section '%s': %s",
                        paper.title,
                        name,
                        text[:100],
                    )
                    continue
                seen_claims.add(text)
                claims.append(
                    Claim(
                        text=text,
                        paper_id=paper_id,
                        section_name=name,
                    )
                )

    # Cap claims at config limit
    if len(claims) > config.max_claims_per_paper:
        _logger.warning(
            "Capping claims for paper '%s' from %d to %d",
            paper.title,
            len(claims),
            config.max_claims_per_paper,
        )
        claims = claims[: config.max_claims_per_paper]

    profile = PaperProfile(
        ranked_paper=ranked_paper,
        sections=sections,
        claims=claims,
    )
    return profile, None
