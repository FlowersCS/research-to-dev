"""Unit tests for HttpxScraper (REQ-07)."""

from __future__ import annotations

import httpx
import pytest
import respx

from research_to_dev.extraction.scraper import HttpxScraper


SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Hello World</h1>
    <p>This is a paragraph with <b>bold</b> text.</p>
    <script>console.log("should be removed");</script>
    <style>.hidden { display: none; }</style>
    <nav><a href="/">Home</a></nav>
    <footer>Copyright 2025</footer>
    <p>Another paragraph.</p>
</body>
</html>"""


class TestHttpxScraperFetch:
    """Verify fetch() HTTP behavior."""

    @pytest.mark.asyncio
    async def test_successful_fetch_returns_html(self) -> None:
        """HTTP 200 returns raw HTML string."""
        with respx.mock:
            respx.get("https://example.com/page").mock(
                return_value=httpx.Response(200, text=SAMPLE_HTML)
            )
            scraper = HttpxScraper()
            html = await scraper.fetch("https://example.com/page")
            assert "<h1>Hello World</h1>" in html

    @pytest.mark.asyncio
    async def test_http_404_raises(self) -> None:
        """HTTP 404 raises HTTPStatusError."""
        with respx.mock:
            respx.get("https://example.com/missing").mock(
                return_value=httpx.Response(404)
            )
            scraper = HttpxScraper()
            with pytest.raises(httpx.HTTPStatusError):
                await scraper.fetch("https://example.com/missing")

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        """Timeout raises TimeoutException."""
        with respx.mock:
            respx.get("https://example.com/slow").mock(
                side_effect=httpx.TimeoutException("Connection timed out")
            )
            scraper = HttpxScraper()
            with pytest.raises(httpx.TimeoutException):
                await scraper.fetch("https://example.com/slow")

    @pytest.mark.asyncio
    async def test_http_500_raises(self) -> None:
        """HTTP 500 raises HTTPStatusError."""
        with respx.mock:
            respx.get("https://example.com/error").mock(
                return_value=httpx.Response(500)
            )
            scraper = HttpxScraper()
            with pytest.raises(httpx.HTTPStatusError):
                await scraper.fetch("https://example.com/error")


class TestHttpxScraperLifecycle:
    """Verify client lifecycle behavior."""

    @pytest.mark.asyncio
    async def test_aclose_closes_owned_client(self) -> None:
        scraper = HttpxScraper()
        client = scraper._client

        assert client.is_closed is False
        await scraper.aclose()
        assert client.is_closed is True

    @pytest.mark.asyncio
    async def test_aclose_does_not_close_injected_client(self) -> None:
        client = httpx.AsyncClient()
        scraper = HttpxScraper(client=client)

        await scraper.aclose()
        assert client.is_closed is False
        await client.aclose()

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_owned_client(self) -> None:
        scraper = HttpxScraper()
        client = scraper._client

        async with scraper as managed:
            assert managed is scraper
            assert client.is_closed is False

        assert client.is_closed is True

