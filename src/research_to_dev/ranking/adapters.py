"""OpenAI adapters for embedding and LLM judgment.

Follows the same constructor-injection pattern as ``HttpxScraper``:
dependencies (``AsyncOpenAI`` client, model, logger) are received via
``__init__``. No env-var coupling inside the adapter.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI


class OpenAIEmbedder:
    """Embedder implementation using OpenAI ``text-embedding-3-small``.

    Satisfies the ``Embedder`` Protocol.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str = "text-embedding-3-small",
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model
        self._logger = logger or logging.getLogger(__name__)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single API call.

        Returns one embedding vector per input text, in the same order.
        """
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [d.embedding for d in response.data]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Delegates to ``embed()`` with a single-element list for consistency.
        """
        results = await self.embed([query])
        return results[0]


class OpenAIJudge:
    """LLM judge implementation using OpenAI ``gpt-4o-mini``.

    Uses a single batched prompt (AD-09): all top-K papers are sent
    in one call with structured JSON output via
    ``response_format={"type": "json_object"}``.

    Satisfies the ``LLMJudge`` Protocol.
    """

    SYSTEM_PROMPT = (
        "You are a research paper relevance judge. Given a research "
        "query and a list of papers, evaluate which papers are most "
        "relevant to the query.\n\n"
        "Return a JSON object with a 'papers' array. Each entry must "
        "have:\n"
        "- paper_index (int): the index of the paper from the input list\n"
        "- relevance_score (float): 0.0 to 1.0, how relevant the paper "
        "is to the query\n"
        "- reasoning (str): brief explanation of why this paper is "
        "relevant (or not relevant if score is low)\n\n"
        "Return only papers you consider at least somewhat relevant "
        "(score > 0). Order by relevance descending."
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

    async def judge(
        self, query: str, papers: list[dict]
    ) -> list[dict]:
        """Judge paper relevance via a single batched LLM call.

        Args:
            query: The research question or topic.
            papers: List of paper dicts with keys
                ``paper_index``, ``title``, ``abstract``, ``semantic_score``.

        Returns:
            List of dicts with keys
            ``paper_index``, ``relevance_score``, ``reasoning``.
        """
        papers_json = json.dumps(papers, indent=2)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Research query: {query}\n\n"
                        f"Papers to evaluate:\n{papers_json}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content is None:
            self._logger.warning("LLM returned empty response content")
            return []

        try:
            result = json.loads(content)
            return result.get("papers", [])
        except json.JSONDecodeError:
            self._logger.warning("Failed to parse LLM JSON response: %s", content[:200])
            return []
