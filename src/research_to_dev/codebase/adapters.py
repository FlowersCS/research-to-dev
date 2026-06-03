"""OpenAI adapter for codebase component description.

Follows the same constructor-injection pattern as ``OpenAIJudge``
and ``OpenAISectionExtractor``: dependencies (``AsyncOpenAI`` client,
model, logger) are received via ``__init__``. No env-var coupling
inside the adapter.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI


class OpenAIDescriber:
    """Component descriptor using OpenAI ``gpt-4o-mini``.

    Sends the full module source code and component list to the LLM
    in a single batched prompt per module (same pattern as
    ``OpenAIJudge.judge()``). Receives structured JSON with
    ``descriptions``, ``module_summary``, and ``key_responsibilities``.

    Satisfies the ``CodebaseDescriber`` Protocol.
    """

    SYSTEM_PROMPT = (
        "You are a codebase analysis assistant. Given the full source "
        "code of a Python module and a list of its components "
        "(functions, classes, methods), produce meaningful "
        "descriptions and a module summary.\n\n"
        "For each component:\n"
        "- If the component already has a useful, non-trivial "
        "docstring, preserve it as the description. A docstring like "
        "\"Constructor.\" or \"Initialize self.\" is trivial — replace "
        "it.\n"
        "- If the component has no docstring or a trivial one, "
        "generate a concise 1-2 sentence description based on its "
        "name, signature, and the surrounding code context.\n\n"
        "For the module:\n"
        "- Provide a brief summary (2-3 sentences) describing the "
        "module's architectural role and purpose.\n"
        "- List 2-5 key responsibilities this module fulfills.\n\n"
        "Return a JSON object with exactly this structure:\n"
        '{\n'
        '  "descriptions": [\n'
        '    {"name": "component_name", "kind": "function|class|method", "description": "What it does."}\n'
        '  ],\n'
        '  "module_summary": "Brief module description.",\n'
        '  "key_responsibilities": ["Responsibility 1", "Responsibility 2"]\n'
        '}\n\n'
        "The 'descriptions' list must have one entry for every "
        "component in the input list. Do not omit any component."
    )

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        *,
        model: str = "gpt-4o-mini",
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model
        self._logger = logger or logging.getLogger(__name__)

    async def describe(
        self,
        module_path: str,
        source: str,
        components: list[dict],
    ) -> dict:
        """Generate descriptions for components and a module summary.

        Sends the full module source and component list in a single
        batched LLM call. Returns structured JSON matching the
        ``CodebaseDescriber`` Protocol contract.

        Args:
            module_path: Dotted module name or file path for context.
            source: Full source code of the module.
            components: List of raw component dicts with keys
                ``name``, ``kind``, ``signature``, ``docstring``.

        Returns:
            Dict with keys ``descriptions``, ``module_summary``,
            ``key_responsibilities``. On any failure (CA-07) returns
            a degraded dict with empty descriptions.
        """
        # Build a compact component list for the prompt
        # Preserve existing useful docstrings (CA-06)
        component_info: list[dict] = []
        for c in components:
            info = {
                "name": c.get("name", ""),
                "kind": c.get("kind", "function"),
                "signature": c.get("signature", ""),
            }
            docstring = c.get("docstring")
            if docstring and not _is_trivial_docstring(docstring):
                info["existing_docstring"] = docstring
            component_info.append(info)

        components_json = json.dumps(component_info, indent=2)

        user_message = (
            f"Module: {module_path}\n\n"
            f"Source code:\n```python\n{source}\n```\n\n"
            f"Components to describe:\n{components_json}"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self._logger.warning(
                "LLM call failed for module '%s': %s", module_path, exc
            )
            return _degraded_result(components)

        content = response.choices[0].message.content
        if content is None:
            self._logger.warning(
                "LLM returned empty response content for module '%s'",
                module_path,
            )
            return _degraded_result(components)

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            self._logger.warning(
                "Failed to parse LLM JSON response for module '%s': %s",
                module_path,
                content[:200],
            )
            return _degraded_result(components)

        # Validate the top-level structure
        if not isinstance(result, dict):
            self._logger.warning(
                "LLM response is not a dict for module '%s': %s",
                module_path,
                content[:200],
            )
            return _degraded_result(components)

        # Ensure all expected keys exist
        if "descriptions" not in result:
            result["descriptions"] = []
        if "module_summary" not in result:
            result["module_summary"] = ""
        if "key_responsibilities" not in result:
            result["key_responsibilities"] = []

        return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _is_trivial_docstring(docstring: str) -> bool:
    """Check if a docstring is too trivial to be useful.

    Single-word or very short docstrings like "Constructor." or
    "Initialize self." are considered trivial.

    Args:
        docstring: The docstring text to check.

    Returns:
        ``True`` if the docstring is too trivial to use as-is.
    """
    stripped = docstring.strip()
    if len(stripped) < 10:
        return True
    trivial_patterns = (
        "constructor.",
        "initialize self.",
        "initialise self.",
    )
    if stripped.lower().startswith(trivial_patterns):
        return True
    return False


def _degraded_result(components: list[dict]) -> dict:
    """Build a degraded result dict with empty descriptions (CA-07).

    Each component gets an empty description so the pipeline can
    continue without crashing.
    """
    return {
        "descriptions": [
            {
                "name": c.get("name", ""),
                "kind": c.get("kind", "function"),
                "description": c.get("docstring") or "",
            }
            for c in components
        ],
        "module_summary": "",
        "key_responsibilities": [],
    }
