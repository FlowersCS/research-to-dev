"""OpenAI adapter for hypothesis generation and reflection.

Follows the same constructor-injection pattern as ``OpenAIJudge``,
``OpenAISectionExtractor``, and ``OpenAICorrelationClassifier``:
dependencies (``AsyncOpenAI`` client, model, logger) are received via
``__init__``. No env-var coupling inside the adapter.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI


class OpenAIHypothesisGenerator:
    """Hypothesis generator and critic using OpenAI ``gpt-4o-mini``.

    Satisfies the ``HypothesisGenerator`` Protocol with two methods:
    ``generate()`` for initial hypothesis creation and
    ``critique_and_expand()`` for the reflection loop (AD-08).
    """

    GENERATE_SYSTEM_PROMPT = (
        "You are a research-to-code hypothesis generator. "
        "Your task is to analyze relationships between research paper "
        "claims and codebase components, then generate specific, "
        "actionable hypotheses for code improvements.\n\n"
        "For each hypothesis, provide:\n"
        '- "title": concise, specific title (5-10 words)\n'
        '- "description": detailed explanation (2-4 sentences)\n'
        '- "approach": specific technical approach to implement or test '
        "(1-3 sentences)\n"
        '- "supporting_papers": list of paper titles (strings)\n'
        '- "target_metric": what metric would improve\n'
        '- "expected_improvement": qualitative description of expected '
        "improvement\n"
        '- "code_changes": what specific code files/components would '
        "change\n"
        '- "scores": object with "relevance", "feasibility", '
        '"evidence" (each 1-10 integers)\n'
        '- "correlations": list of correlation IDs that ground this '
        "hypothesis\n"
        '- "success_criteria": measurable criteria to validate\n\n'
        "Scores guidance:\n"
        "- relevance: How relevant is this hypothesis to the research? "
        "(1=irrelevant, 10=directly addresses claims)\n"
        "- feasibility: How feasible to implement? "
        "(1=requires major rewrite, 10=minor tweak)\n"
        "- evidence: How strong is the evidence from papers? "
        "(1=speculative, 10=strong empirical support)\n\n"
        "Return a JSON object with a 'hypotheses' array."
    )

    CRITIQUE_SYSTEM_PROMPT = (
        "You are a research-to-code hypothesis critic and generator. "
        "Given existing hypotheses about code improvements, your task "
        "is to:\n"
        "1. Critique each hypothesis for weaknesses, gaps, or "
        "unrealistic assumptions\n"
        "2. Generate new hypotheses that fill identified gaps or "
        "address weaknesses\n\n"
        "For critiques, focus on:\n"
        "- Missing evidence or weak claims\n"
        "- Implementation feasibility concerns\n"
        "- Overlooked alternative approaches\n"
        "- Unrealistic expected improvements\n\n"
        "For new hypotheses, follow this format:\n"
        '- "title": concise specific title\n'
        '- "description": detailed explanation\n'
        '- "approach": specific technical approach\n'
        '- "supporting_papers": list of paper titles\n'
        '- "target_metric": what metric would improve\n'
        '- "expected_improvement": qualitative improvement description\n'
        '- "code_changes": specific code files/components\n'
        '- "scores": object with "relevance", "feasibility", '
        '"evidence" (each 1-10 integers)\n'
        '- "correlations": list of correlation IDs\n'
        '- "success_criteria": measurable criteria\n\n'
        "Return a JSON object with:\n"
        '- "critiques": array of critique objects, each with '
        '"hypothesis_title" (string) and "weaknesses" (list of strings)\n'
        '- "new_hypotheses": array of new hypothesis objects'
    )

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str = "gpt-4o-mini",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the generator with optional injected dependencies.

        Args:
            client: AsyncOpenAI instance. A default is created if not
                supplied.
            model: OpenAI chat model to use (default ``gpt-4o-mini``).
            logger: Logger instance. Defaults to
                ``logging.getLogger(__name__)``.
        """
        self._client = client or AsyncOpenAI()
        self._model = model
        self._logger = logger or logging.getLogger(__name__)

    async def generate(
        self,
        title: str,
        paper_claims: str,
        correlations: list[dict],
        query: str,
        round_num: int = 0,
        general_content: str | None = None,
    ) -> list[dict]:
        """Generate initial hypotheses for a paper-cluster (AD-08, D2).

        Args:
            title: Paper title for context.
            paper_claims: Concatenated claim texts from this paper.
            correlations: List of correlation dicts with keys
                ``id``, ``claim_text``, ``target_name``,
                ``target_type``, ``target_description``,
                ``correlation_type``, ``reasoning``,
                ``similarity_score``, ``paper_title``.
            query: Original user query.
            round_num: 0 for initial generation, 1-3 for reflection.
            general_content: Optional supplementary implementation
                context from tutorials, blogs, docs (formatted string).
                Included as ``Implementation References`` section when
                provided.

        Returns:
            List of hypothesis dicts. Empty list on any failure.
        """
        correlations_json = json.dumps(correlations, indent=2)

        general_section = ""
        if general_content:
            general_section = (
                f"Implementation References (tutorials, docs, blogs — "
                f"supplementary HOW context):\n{general_content}\n\n"
            )

        user_prompt = (
            f"Paper Title: {title}\n\n"
            f"Key Claims from this paper:\n{paper_claims}\n\n"
            f"Correlations (claim-to-code mappings):\n"
            f"{correlations_json}\n\n"
            f"{general_section}"
            f"User Query: {query}\n\n"
            f"Generate code improvement hypotheses based on these "
            f"correlations. Each hypothesis MUST reference at least one "
            f"correlation ID from the list above."
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.GENERATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self._logger.warning(
                "LLM generate() failed for paper '%s': %s", title, exc,
            )
            return []

        return self._parse_hypotheses_response(response, title)

    async def critique_and_expand(
        self,
        title: str,
        paper_claims: str,
        correlations: list[dict],
        existing_hypotheses: list[dict],
        query: str,
        round_num: int,
        general_content: str | None = None,
    ) -> dict:
        """Critique existing hypotheses and generate new ones (AD-08, D3).

        Args:
            title: Paper title for context.
            paper_claims: Concatenated claim texts from this paper.
            correlations: List of correlation dicts (same structure as
                ``generate()``).
            existing_hypotheses: Current hypotheses as dicts (from all
                papers — shared for cross-paper critique).
            query: Original user query.
            round_num: Current reflection round (1-3).
            general_content: Optional supplementary implementation
                context from tutorials, blogs, docs (formatted string).

        Returns:
            Dict with keys ``critiques`` (list of critique dicts) and
            ``new_hypotheses`` (list of hypothesis dicts). Returns
            ``{"critiques": [], "new_hypotheses": []}`` on any failure.
        """
        correlations_json = json.dumps(correlations, indent=2)
        hypotheses_json = json.dumps(existing_hypotheses, indent=2)

        general_section = ""
        if general_content:
            general_section = (
                f"Implementation References (tutorials, docs, blogs — "
                f"supplementary HOW context):\n{general_content}\n\n"
            )

        user_prompt = (
            f"Paper Title: {title}\n\n"
            f"Key Claims from this paper:\n{paper_claims}\n\n"
            f"Correlations (claim-to-code mappings):\n"
            f"{correlations_json}\n\n"
            f"Existing Hypotheses (from all papers):\n"
            f"{hypotheses_json}\n\n"
            f"{general_section}"
            f"User Query: {query}\n\n"
            f"Reflection Round: {round_num}\n\n"
            f"Critique each existing hypothesis for weaknesses, then "
            f"generate new hypotheses filling identified gaps. "
            f"Each new hypothesis MUST reference at least one correlation "
            f"ID from the correlations above."
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.CRITIQUE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self._logger.warning(
                "LLM critique_and_expand() failed for paper '%s' "
                "round %d: %s",
                title, round_num, exc,
            )
            return {"critiques": [], "new_hypotheses": []}

        content = response.choices[0].message.content
        if content is None:
            self._logger.warning(
                "LLM returned empty response for critique paper '%s' "
                "round %d",
                title, round_num,
            )
            return {"critiques": [], "new_hypotheses": []}

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            self._logger.warning(
                "Failed to parse critique JSON for paper '%s' round %d: %s",
                title, round_num, content[:200],
            )
            return {"critiques": [], "new_hypotheses": []}

        if not isinstance(result, dict):
            self._logger.warning(
                "Critique response not a dict for paper '%s' round %d",
                title, round_num,
            )
            return {"critiques": [], "new_hypotheses": []}

        critiques = result.get("critiques", [])
        new_hypotheses = result.get("new_hypotheses", [])

        if not isinstance(critiques, list):
            critiques = []
        if not isinstance(new_hypotheses, list):
            new_hypotheses = []

        return {"critiques": critiques, "new_hypotheses": new_hypotheses}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_hypotheses_response(
        self,
        response: object,
        title: str,
    ) -> list[dict]:
        """Extract hypotheses list from a generate() LLM response.

        Returns empty list on any parsing failure.
        """
        content = response.choices[0].message.content
        if content is None:
            self._logger.warning(
                "LLM returned empty response for paper '%s'", title,
            )
            return []

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            self._logger.warning(
                "Failed to parse LLM JSON for paper '%s': %s",
                title, content[:200],
            )
            return []

        if not isinstance(result, dict):
            self._logger.warning(
                "LLM response not a dict for paper '%s': %s",
                title, content[:200],
            )
            return []

        hypotheses = result.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            self._logger.warning(
                "'hypotheses' field not a list for paper '%s'", title,
            )
            return []

        return hypotheses
