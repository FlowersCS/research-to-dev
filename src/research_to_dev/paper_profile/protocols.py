"""Protocols for pluggable paper profiling dependencies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SectionExtractor(Protocol):
    """Async protocol for section and claim extraction.

    Implementations receive a paper title and abstract text and return
    a raw dict with the expected JSON schema. Currently implemented by
    ``OpenAISectionExtractor`` using ``gpt-4o-mini``.
    """

    async def extract(self, title: str, abstract: str) -> dict:
        """Extract sections and claims from paper text in one LLM call.

        Args:
            title: Paper title.
            abstract: Paper abstract text (may be empty string).

        Returns:
            Dict with shape ``{'sections': [{name, content, claims: [{text}]}]}``.
        """
        ...
