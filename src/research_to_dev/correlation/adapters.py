"""OpenAI adapter for correlation classification.

Follows the same constructor-injection pattern as ``OpenAIJudge``,
``OpenAISectionExtractor``, and ``OpenAIDescriber``: dependencies
(``AsyncOpenAI`` client, model, logger) are received via ``__init__``.
No env-var coupling inside the adapter.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI


class OpenAICorrelationClassifier:
    """Correlation classifier using OpenAI ``gpt-4o-mini``.

    Sends all candidate matches for one paper in a single batched
    prompt (AD-02, D9). Receives structured JSON with classified
    correlation types and reasoning (D2, CM-03).

    Satisfies the ``CorrelationClassifier`` Protocol.
    """

    SYSTEM_PROMPT = (
        "You are a research-to-code correlation classifier. "
        "Your task is to determine whether claims from a research paper "
        "describe techniques, methods, algorithms, or results that have "
        "a meaningful correlation with specific code components or modules.\n\n"
        "For each candidate match, classify the correlation type:\n"
        '- "direct_solution": the code directly implements the technique or method described in the claim\n'
        '- "related_technique": the code uses a technique related to, but different from, the claim\n'
        '- "contradicts": the claim describes an approach that contradicts or invalidates the code\'s design\n'
        '- "prerequisite": the claim describes foundational work that the code builds upon\n\n'
        "If a candidate does NOT represent a real correlation, "
        "do NOT include it in the output — only include matches you "
        "believe are genuine correlations.\n\n"
        "Return a JSON object with a 'matches' array. Each entry MUST have:\n"
        '- "claim_text": the original claim text (string)\n'
        '- "target_name": the name of the code component or module (string)\n'
        '- "correlation_type": one of direct_solution, related_technique, contradicts, prerequisite\n'
        '- "reasoning": a brief 1-2 sentence explanation of why this correlation exists\n'
    )

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str = "gpt-4o-mini",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the classifier with optional injected dependencies.

        Args:
            client: AsyncOpenAI instance. A default is created if not supplied.
            model: OpenAI chat model to use (default ``gpt-4o-mini``).
            logger: Logger instance. Defaults to ``logging.getLogger(__name__)``.
        """
        self._client = client or AsyncOpenAI()
        self._model = model
        self._logger = logger or logging.getLogger(__name__)

    async def classify(
        self,
        title: str,
        paper_claims: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Classify correlation candidates for a single paper.

        Submits one batched LLM call with all candidates. Only
        LLM-validated matches are returned (D10 — rejects discarded
        silently by caller).

        Args:
            title: Paper title for context.
            paper_claims: Concatenated claim texts from this paper,
                formatted for the prompt.
            candidates: List of candidate dicts with keys
                ``claim_text``, ``similarity_score``, ``target_text``,
                ``target_type`` (``'component'`` or ``'module'``),
                ``target_name``, ``module_name``.

        Returns:
            List of validated match dicts with keys
            ``claim_text``, ``target_name``, ``correlation_type``,
            ``reasoning``. Empty list on any failure (CM-05).
        """
        if not candidates:
            return []

        candidates_json = json.dumps(candidates, indent=2)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Paper title: {title}\n\n"
                            f"Key claims from this paper:\n{paper_claims}\n\n"
                            f"Candidate code matches (already filtered by "
                            f"embedding similarity — verify each one):\n"
                            f"{candidates_json}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self._logger.warning(
                "LLM classification failed for paper '%s': %s", title, exc
            )
            return []

        content = response.choices[0].message.content
        if content is None:
            self._logger.warning(
                "LLM returned empty response content for paper '%s'", title
            )
            return []

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            self._logger.warning(
                "Failed to parse LLM JSON response for paper '%s': %s",
                title,
                content[:200],
            )
            return []

        if not isinstance(result, dict):
            self._logger.warning(
                "LLM response is not a dict for paper '%s': %s",
                title,
                content[:200],
            )
            return []

        matches = result.get("matches", [])
        if not isinstance(matches, list):
            self._logger.warning(
                "LLM 'matches' field is not a list for paper '%s'", title
            )
            return []

        return matches
