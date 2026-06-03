"""Protocols for pluggable correlation pipeline dependencies.

A new ``CorrelationClassifier`` Protocol is introduced because the
classification task (claim↔code matching + type assignment) differs
from the ranking funnel's ``LLMJudge`` (relevance scoring).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CorrelationClassifier(Protocol):
    """Async protocol for LLM-based correlation classification (AD-02, D12).

    Receives one paper's candidates in a single batched call so the
    LLM can leverage cross-claim context (D9). Returns only LLM-validated
    matches — rejected candidates are discarded by the caller (D10).

    Implemented by ``OpenAICorrelationClassifier`` using ``gpt-4o-mini``
    with ``response_format={"type": "json_object"}``.
    """

    async def classify(  # noqa: D102
        self,
        title: str,
        paper_claims: str,
        candidates: list[dict],
    ) -> list[dict]:
        ...
