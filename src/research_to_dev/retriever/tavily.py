"""Tavily retriever — queries the Tavily web search API."""

from __future__ import annotations

import logging
from typing import Literal

import httpx

from research_to_dev.retriever.protocol import SearchResult

_TAVILY_API = "https://api.tavily.com/search"


class TavilyRetriever:
    """Async retriever for Tavily's general web search API.

    Requires an API key (Bearer auth). Returns web results
    normalized to SearchResult — academic fields (authors,
    abstract, published_date) are set to None since Tavily
    provides content-based results.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        api_key: str,
        logger: logging.Logger | None = None,
        timeout: float = 30.0,
        search_depth: Literal["basic", "advanced"] = "basic",
    ) -> None:
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._api_key = api_key
        self._logger = logger or logging.getLogger(__name__)
        self._timeout = timeout
        self._search_depth = search_depth

    async def retrieve(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        """Query Tavily and return normalized SearchResults.

        Returns empty list on any error — never raises into caller code.
        """
        if not self._api_key:
            self._logger.error(
                "Tavily API key is empty or missing — cannot query."
            )
            return []

        try:
            headers = {"Authorization": f"Bearer {self._api_key}"}
            payload = {
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": self._search_depth,
            }
            response = await self._client.post(
                _TAVILY_API, json=payload, headers=headers
            )

            if response.status_code == 432:
                self._logger.warning(
                    "Tavily plan limit reached (432) for query=%r", query
                )
                return []

            response.raise_for_status()
            return self._parse(response.json())
        except Exception as exc:
            self._logger.error(
                "Tavily retrieval failed for query=%r: %s", query, exc
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(data: dict) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in data.get("results", []):
            metadata: dict = {}
            if item.get("raw_content"):
                metadata["raw_content"] = item["raw_content"]
            if item.get("score") is not None:
                metadata["score"] = item["score"]

            results.append(
                SearchResult(
                    source="tavily",
                    title=item.get("title") or "Untitled",
                    url=item.get("url"),
                    abstract=item.get("content"),
                    authors=None,
                    published_date=None,
                    metadata=metadata,
                )
            )
        return results
