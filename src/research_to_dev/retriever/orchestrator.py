"""Retriever orchestration — parallel multi-source search facade."""

from __future__ import annotations

import asyncio
import logging
import time

from research_to_dev.retriever.protocol import (
    RetrievalError,
    RetrievalResult,
    Retriever,
    SearchResult,
)


class RetrieverOrchestrator:
    """Runs multiple retriever adapters in parallel and aggregates results.

    Uses ``asyncio.gather(return_exceptions=True)`` so one failing
    adapter never blocks others. The result is a ``RetrievalResult``
    with all successful items merged and one ``RetrievalError`` per
    failed adapter.
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        logger: logging.Logger | None = None,
        verbose: bool = False,
    ) -> None:
        self._retrievers = retrievers
        self._logger = logger or logging.getLogger(__name__)
        self._verbose = verbose

    async def search(self, query: str, max_results: int = 10) -> RetrievalResult:
        """Fan out to all adapters concurrently and aggregate.

        Returns:
            RetrievalResult with merged successful results and
            one RetrievalError per failed adapter.
        """
        all_results: list[SearchResult] = []
        all_errors: list[RetrievalError] = []

        tasks = [self._timed_retrieve(r, query, max_results) for r in self._retrievers]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for retriever, outcome in zip(self._retrievers, outcomes):
            if isinstance(outcome, Exception):
                src = type(retriever).__name__
                all_errors.append(
                    RetrievalError(
                        source=src,
                        message=str(outcome),
                        exception_type=type(outcome).__name__,
                    )
                )
                self._logger.warning("Retriever %s failed: %s", src, outcome)
            else:
                all_results.extend(outcome)

        if self._verbose:
            total = len(all_results)
            errors_count = len(all_errors)
            self._logger.info(
                "Search complete: %d results, %d errors (total adapters: %d)",
                total,
                errors_count,
                len(self._retrievers),
            )

        return RetrievalResult(results=all_results, errors=all_errors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _timed_retrieve(
        self,
        retriever: Retriever,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Wrap retrieve() with optional verbose timing."""
        if self._verbose:
            t0 = time.monotonic()
            results = await retriever.retrieve(query, max_results)
            elapsed = time.monotonic() - t0
            self._logger.info(
                "%s: %d results in %.2fs",
                type(retriever).__name__,
                len(results),
                elapsed,
            )
            return results
        return await retriever.retrieve(query, max_results)
