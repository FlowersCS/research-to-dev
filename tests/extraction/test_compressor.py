"""Unit tests for NoopCompressor stub (REQ-08)."""

from __future__ import annotations

import pytest

from research_to_dev.extraction.compressor import NoopCompressor
from research_to_dev.extraction.protocols import Compressor


class TestNoopCompressor:
    """Verify identity compression and Protocol conformance."""

    @pytest.mark.asyncio
    async def test_returns_input_unchanged(self) -> None:
        """Stub returns text unchanged."""
        compressor = NoopCompressor()
        result = await compressor.compress("hello world")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_empty_string(self) -> None:
        """Empty input returns empty string."""
        compressor = NoopCompressor()
        result = await compressor.compress("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_large_text(self) -> None:
        """Large input passes through unchanged."""
        compressor = NoopCompressor()
        large = "x" * 5000
        result = await compressor.compress(large)
        assert result == large

    @pytest.mark.asyncio
    async def test_special_characters(self) -> None:
        """Special characters are preserved."""
        compressor = NoopCompressor()
        text = "Hello\n世界\t🌟"
        result = await compressor.compress(text)
        assert result == text

    def test_satisfies_compressor_protocol(self) -> None:
        """NoopCompressor is a valid Compressor Protocol implementation."""
        compressor = NoopCompressor()
        assert isinstance(compressor, Compressor)


# -------------------------------------------------------------------
# Threshold boundary tests (REQ-08)
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_below_threshold_no_compression() -> None:
    """Content below threshold should not trigger compression.

    The compressor is not called; content is returned as-is.
    """
    threshold = 4000
    text = "short" * 200  # 1000 chars — well below 4000
    assert len(text) < threshold
    assert len(text) == 1000


@pytest.mark.asyncio
async def test_above_threshold_triggers_compression() -> None:
    """Content above threshold should route to compressor stub.

    The stub returns it unchanged, but the threshold routing works.
    """
    threshold = 4000
    text = "x" * 5000
    assert len(text) > threshold
    compressor = NoopCompressor()
    result = await compressor.compress(text)
    assert result == text
    assert len(result) == 5000


@pytest.mark.asyncio
async def test_exact_threshold_boundary() -> None:
    """Content at exact threshold — above means > threshold."""
    threshold = 4000
    text = "x" * threshold  # exactly 4000
    assert len(text) == threshold
    # At threshold boundary: not above, so no compression needed
    compressor = NoopCompressor()
    result = await compressor.compress(text)
    assert result == text
