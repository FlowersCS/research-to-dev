"""Content extraction pipeline — transforms search results into typed content.

Public API
----------
- **Pipeline**: ``ExtractionPipeline`` — routes by source, extracts content
- **Data**: ``AcademicContent``, ``GeneralContent``, ``ExtractionResult``, ``ExtractionError``
- **Protocols**: ``Compressor``, ``Scraper`` — pluggable dependencies
- **Stubs**: ``NoopCompressor`` — identity compression for MVP
- **Scraper**: ``HttpxScraper`` — httpx-based HTML fetching
"""

from research_to_dev.extraction.compressor import NoopCompressor
from research_to_dev.extraction.pipeline import ExtractionPipeline
from research_to_dev.extraction.protocols import Compressor, Scraper
from research_to_dev.extraction.scraper import HttpxScraper
from research_to_dev.extraction.types import (
    AcademicContent,
    ExtractionError,
    ExtractionResult,
    GeneralContent,
)

__all__ = [
    "AcademicContent",
    "Compressor",
    "ExtractionError",
    "ExtractionPipeline",
    "ExtractionResult",
    "GeneralContent",
    "HttpxScraper",
    "NoopCompressor",
    "Scraper",
]
