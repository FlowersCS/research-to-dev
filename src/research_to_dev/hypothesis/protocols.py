"""Protocols for pluggable hypothesis generation dependencies.

A new ``HypothesisGenerator`` Protocol is introduced because the
generation task (code improvement hypothesis creation + critique+expand)
differs from the correlation module's ``CorrelationClassifier`` (matching).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HypothesisGenerator(Protocol):
    """Async protocol for LLM-based hypothesis generation and reflection.

    Two methods: ``generate()`` for initial hypothesis creation per
    paper-cluster, and ``critique_and_expand()`` for the reflection loop
    (D2, D3, AD-08).

    Implemented by ``OpenAIHypothesisGenerator`` using ``gpt-4o-mini``
    with ``response_format={"type": "json_object"}``.
    """

    async def generate(  # noqa: D102
        self,
        title: str,
        paper_claims: str,
        correlations: list[dict],
        query: str,
        round_num: int = 0,
        general_content: str | None = None,
    ) -> list[dict]:
        ...

    async def critique_and_expand(  # noqa: D102
        self,
        title: str,
        paper_claims: str,
        correlations: list[dict],
        existing_hypotheses: list[dict],
        query: str,
        round_num: int,
        general_content: str | None = None,
    ) -> dict:
        ...
