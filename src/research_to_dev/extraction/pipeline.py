"""Content extraction pipeline — concurrent routing + compression with partial failure tolerance."""

from __future__ import annotations

import asyncio
import logging

from research_to_dev.extraction.academic import extract_academic_content
from research_to_dev.extraction.general import extract_general_content
from research_to_dev.extraction.protocols import Compressor, Scraper
from research_to_dev.extraction.types import (
    AcademicContent,
    ExtractionError,
    ExtractionResult,
    GeneralContent,
)
from research_to_dev.retriever.protocol import (
    RetrievalResult,
    SearchResult,
)

_ContentType = AcademicContent | GeneralContent


class ExtractionPipeline:
    """Routes SearchResults to the correct extractor and compresses output.

    Uses ``asyncio.gather(return_exceptions=True)`` so one failing
    extraction never blocks others. Exceptions are converted to
    ``ExtractionError`` entries and a degraded content object is still
    produced for the failed item (so ``len(contents) == len(inputs)``).
    """

    def __init__(
        self,
        scraper: Scraper,
        *,
        compressor: Compressor | None = None,
        compress_threshold: int = 4000,
        logger: logging.Logger | None = None,
    ) -> None:
        self._scraper = scraper
        self._compressor = compressor
        self._compress_threshold = compress_threshold
        self._logger = logger or logging.getLogger(__name__)

        self._handlers = {
            "arxiv": self._extract_academic,
            "semantic_scholar": self._extract_academic,
            "tavily": self._extract_general,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, retrieval: RetrievalResult) -> ExtractionResult:
        """Fan out extraction to all results concurrently and aggregate.

        Returns:
            ExtractionResult with typed content objects and any errors.
        """
        all_contents: list[_ContentType] = []
        all_errors: list[ExtractionError] = []

        tasks = [self._extract_one(r) for r in retrieval.results]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for result, outcome in zip(retrieval.results, outcomes):
            if isinstance(outcome, BaseException):
                src = result.source
                all_errors.append(
                    ExtractionError(
                        source=src,
                        url=result.url,
                        message=str(outcome),
                        exception_type=type(outcome).__name__,
                    )
                )
                self._logger.warning(
                    "Extraction failed for %r (%s): %s", result.title, src, outcome
                )
                # Still produce a degraded content so len(contents) == len(inputs).
                all_contents.append(
                    GeneralContent(
                        source="tavily",  # type: ignore[arg-type]  # only tavily can fail
                        title=result.title,
                        url=result.url,
                        body="",
                        metadata=result.metadata or {},
                    )
                )
            else:
                all_contents.append(outcome)

        # Apply compression to bodies that exceed the threshold.
        if self._compressor is not None:
            all_contents = await self._compress_contents(all_contents, self._compressor)

        return ExtractionResult(contents=all_contents, errors=all_errors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _extract_academic(self, result: SearchResult) -> _ContentType:
        """Extract AcademicContent from a SearchResult."""
        return extract_academic_content(result)

    async def _extract_general(self, result: SearchResult) -> _ContentType:
        """Extract GeneralContent from a SearchResult using the scraper."""
        return await extract_general_content(result, scraper=self._scraper)

    async def _extract_one(self, result: SearchResult) -> _ContentType:
        """Route a single SearchResult to the correct extractor via dict dispatch."""
        handler = self._handlers.get(result.source)
        if handler is None:
            raise ValueError(f"Unknown source: {result.source}")
        return await handler(result)

    async def _compress_contents(
        self,
        contents: list[_ContentType],
        compressor: Compressor,
    ) -> list[_ContentType]:
        """Compress GeneralContent bodies that exceed the configured threshold."""
        for content in contents:
            if isinstance(content, GeneralContent):
                body = content.body
                if body and len(body) > self._compress_threshold:
                    content.body = await compressor.compress(body)
        return contents
