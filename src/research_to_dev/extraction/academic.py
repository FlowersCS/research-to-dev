"""Academic content extraction — maps SearchResult → AcademicContent."""

from __future__ import annotations

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.retriever.protocol import SearchResult


def extract_academic_content(result: SearchResult) -> AcademicContent:
    """Map a structured SearchResult to AcademicContent.

    Uses ``abstract`` as the body since academic sources provide
    structured metadata, not raw page content.
    """
    return AcademicContent(
        source=result.source,  # type: ignore[arg-type]
        title=result.title,
        url=result.url,
        abstract=result.abstract,
        authors=result.authors,
        published_date=result.published_date,
        metadata=result.metadata,
    )
