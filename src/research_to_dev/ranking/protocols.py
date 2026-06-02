"""Protocols for pluggable ranking pipeline dependencies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Async protocol for text embedding.

    Implementations must support both single-query and batch-text
    embedding. Currently implemented by ``OpenAIEmbedder`` using
    ``text-embedding-3-small``.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:  # noqa: D102
        ...

    async def embed_query(self, query: str) -> list[float]:  # noqa: D102
        ...


@runtime_checkable
class LLMJudge(Protocol):
    """Async protocol for LLM-based relevance judgment.

    Uses a single batched prompt — all papers are sent in one call
    (AD-09). The implementation returns structured JSON that the
    pipeline maps back to ``AcademicContent`` objects.

    Currently implemented by ``OpenAIJudge`` using ``gpt-4o-mini``.
    """

    async def judge(  # noqa: D102
        self, query: str, papers: list[dict]
    ) -> list[dict]:
        ...
