"""Protocols for pluggable extraction pipeline dependencies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Compressor(Protocol):
    """Async protocol for content compression.

    Implementations may be identity (MVP), LLM-based, or heuristics.
    """

    async def compress(self, text: str) -> str: ...  # noqa: D107


@runtime_checkable
class Scraper(Protocol):
    """Async protocol for fetching raw HTML from a URL.

    Implementations may use httpx, Playwright, etc.
    """

    async def fetch(self, url: str) -> str: ...  # noqa: D107
