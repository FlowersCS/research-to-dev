"""OpenAI adapter for section and claim extraction.

Follows the same constructor-injection pattern as ``OpenAIJudge``:
dependencies (``AsyncOpenAI`` client, model, logger) are received via
``__init__``. No env-var coupling inside the adapter.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI


class OpenAISectionExtractor:
    """Section and claim extractor using OpenAI ``gpt-4o-mini``.

    Uses a single LLM call per paper (AD-03): sections and claims are
    extracted together in one structured JSON response via
    ``response_format={"type": "json_object"}``.

    Satisfies the ``SectionExtractor`` Protocol.
    """

    SYSTEM_PROMPT = (
        "You are a research paper section and claim extractor. "
        "Given a paper title and abstract, extract the paper's "
        "sections and key assertions.\n\n"
        "Return a JSON object with a 'sections' array. Each entry "
        "must have:\n"
        '- "name" (str): the section label — use "abstract", '
        '"methods", "results", "conclusion", or similar\n'
        '- "content" (str): the text content of that section\n'
        '- "claims" (list): an array of assertion-level claims '
        "found in that section. Each claim object has:\n"
        '  - "text" (str): a concise, verifiable statement '
        "(e.g. 'Model X achieves 94.2% accuracy on benchmark Y')\n\n"
        "If only the abstract is available, extract what you can. "
        "If no claims can be identified, return empty sections.\n\n"
        "Be precise. Claims should be assertion-level — statements "
        "that could be verified or falsified."
    )

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str = "gpt-4o-mini",
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model
        self._logger = logger or logging.getLogger(__name__)

    async def extract(self, title: str, abstract: str) -> dict:
        """Extract sections and claims from a paper in one LLM call.

        Args:
            title: Paper title.
            abstract: Paper abstract text (may be empty string).

        Returns:
            Dict with shape ``{'sections': [{name, content, claims: [{text}]}]}``.
            Returns ``{'sections': []}`` on any failure (RF-05).
        """
        # Build the user message
        text_block = abstract.strip() if abstract else "(no text available)"
        user_message = (
            f"Paper title: {title}\n\n"
            f"Paper text:\n{text_block}"
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
                "LLM call failed for paper '%s': %s", title, exc
            )
            return {"sections": []}

        content = response.choices[0].message.content
        if content is None:
            self._logger.warning(
                "LLM returned empty response content for paper '%s'", title
            )
            return {"sections": []}

        try:
            result = json.loads(content)
            # Validate top-level structure
            if not isinstance(result, dict) or "sections" not in result:
                self._logger.warning(
                    "LLM response missing 'sections' key for paper '%s': %s",
                    title,
                    content[:200],
                )
                return {"sections": []}
            return result
        except json.JSONDecodeError:
            self._logger.warning(
                "Failed to parse LLM JSON response for paper '%s': %s",
                title,
                content[:200],
            )
            return {"sections": []}
