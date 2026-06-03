"""Protocols for pluggable codebase analysis dependencies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CodebaseDescriber(Protocol):
    """Async protocol for LLM-based component description and module summarization.

    Implementations receive the full module source code and a list of
    raw components (dicts with ``name``, ``kind``, ``signature``,
    ``docstring``) and return structured descriptions plus a module
    summary.

    Currently implemented by ``OpenAIDescriber`` using ``gpt-4o-mini``.
    """

    async def describe(  # noqa: D102
        self, module_path: str, source: str, components: list[dict]
    ) -> dict:
        ...
