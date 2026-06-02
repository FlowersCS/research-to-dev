"""Compression stubs — identity pass-through for MVP."""

from __future__ import annotations


class NoopCompressor:
    """Identity compressor — returns text unchanged.

    Serves as the MVP placeholder. When LLM compression is implemented,
    a real compressor replaces this one via the Compressor Protocol.
    """

    async def compress(self, text: str) -> str:
        """Return text unchanged (identity pass-through)."""
        return text
