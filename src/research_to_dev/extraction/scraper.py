"""HTML scraper — fetches pages via httpx."""

from __future__ import annotations

import logging
from types import TracebackType

import httpx


class HttpxScraper:
    """Async scraper that fetches a URL and returns the raw HTML text."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        logger: logging.Logger | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=timeout)
        self._logger = logger or logging.getLogger(__name__)
        self._timeout = timeout

    async def fetch(self, url: str) -> str:
        """Fetch a URL and return the raw HTML body text.

        Raises:
            httpx.TimeoutException: On timeout.
            httpx.HTTPStatusError: On 4xx/5xx responses.
        """
        response = await self._client.get(url)
        response.raise_for_status()
        return response.text

    async def aclose(self) -> None:
        """Close the underlying client if this scraper created it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> HttpxScraper:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

