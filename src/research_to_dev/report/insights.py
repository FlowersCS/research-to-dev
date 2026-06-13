"""Insights generation — LLM-based cross-hypothesis analysis.

Produces insights, recommendations, and evidence correlations by
analysing compiled experiment reports across multiple hypotheses.

Defines the InsightsGenerator Protocol for pluggable
implementations (e.g., OpenAIInsightsGenerator) and the data
model dataclasses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from research_to_dev.report.compilation import HypothesisSummary


# ===================================================================
# Domain Dataclasses
# ===================================================================


@dataclass
class PatternInsight:
    """A detected pattern spanning multiple hypotheses."""

    title: str
    description: str
    affected_hypotheses: list[str]
    confidence: str  # "high" | "medium" | "low"


@dataclass
class Recommendation:
    """An actionable suggestion derived from cross-hypothesis analysis."""

    action: str
    priority: str  # "high" | "medium" | "low"
    rationale: str
    supporting_evidence: list[str]


@dataclass
class EvidenceCorrelation:
    """Maps papers/evidence to specific code areas via hypotheses."""

    paper_id: str
    evidence_summary: str
    related_hypotheses: list[str]
    code_areas: list[str]


@dataclass
class ProgramContext:
    """Per-hypothesis context extracted from program.md."""

    hypothesis_id: str
    direction: str
    baseline: str  # dict serialized as string like "val_accuracy=0.72"
    success_criteria: str


@dataclass
class InsightsReport:
    """Top-level container for LLM-generated cross-hypothesis insights."""

    patterns: list[PatternInsight] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    evidence_correlation: list[EvidenceCorrelation] = field(default_factory=list)


# ===================================================================
# InsightsGenerator Protocol
# ===================================================================


@runtime_checkable
class InsightsGenerator(Protocol):
    """Async protocol for LLM-based cross-hypothesis insights generation."""

    async def generate(
        self,
        hypotheses: list[HypothesisSummary],
        program_contexts: list[ProgramContext] | None = None,
    ) -> InsightsReport | None:
        """Generate cross-hypothesis insights from compiled experiment results.

        Args:
            hypotheses: Compiled per-hypothesis summaries with iteration
                results, status, and baseline comparisons.
            program_contexts: Per-hypothesis context from program.md
                (direction, baseline, success criteria). Optional —
                insights generation proceeds with just hypothesis data
                when absent.

        Returns:
            ``InsightsReport`` on success, ``None`` on any failure
            (LLM error, JSON parse failure, malformed response).
        """
        ...


# ===================================================================
# OpenAIInsightsGenerator Adapter
# ===================================================================


class OpenAIInsightsGenerator:
    """Insights generator using OpenAI ``gpt-4o-mini``.

    Follows the same constructor-injection pattern as
    ``OpenAIHypothesisGenerator``: ``AsyncOpenAI`` client, model, and
    logger are received via ``__init__``. No env-var coupling inside
    the adapter.

    Satisfies the ``InsightsGenerator`` Protocol with a single
    ``generate()`` method that produces cross-hypothesis insights in
    one LLM call.
    """

    INSIGHTS_SYSTEM_PROMPT = (
        "You are a research-to-code insights analyst. "
        "Your task is to analyze compiled experiment results across "
        "multiple hypotheses and produce three types of insights:\n\n"
        "1. **Patterns**: Recurring themes or behaviors that span "
        "multiple hypotheses. Each pattern MUST reference specific "
        "hypothesis IDs.\n\n"
        "2. **Recommendations**: Prioritized, actionable suggestions "
        "for the developer. Each recommendation MUST cite supporting "
        "evidence from specific hypotheses.\n\n"
        "3. **Evidence Correlations**: Connections between research "
        "papers and code areas, linked through the hypotheses that "
        "bridge them.\n\n"
        "Output strict JSON with this schema:\n"
        "{\n"
        '  "patterns": [\n'
        "    {\n"
        '      "title": "string — concise pattern title (5-10 words)",\n'
        '      "description": "string — 2-4 sentence explanation",\n'
        '      "affected_hypotheses": ["string — hypothesis ID", ...],\n'
        '      "confidence": "string — high|medium|low"\n'
        "    }\n"
        "  ],\n"
        '  "recommendations": [\n'
        "    {\n"
        '      "action": "string — specific action to take",\n'
        '      "priority": "string — high|medium|low",\n'
        '      "rationale": "string — why this matters",\n'
        '      "supporting_evidence": '
        '["string — hypothesis ID or paper ref", ...]\n'
        "    }\n"
        "  ],\n"
        '  "evidence_correlation": [\n'
        "    {\n"
        '      "paper_id": "string — paper or source identifier",\n'
        '      "evidence_summary": "string — what the evidence says",\n'
        '      "related_hypotheses": ["string — hypothesis ID", ...],\n'
        '      "code_areas": ["string — file or component", ...]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "IMPORTANT: Every pattern MUST include at least one "
        "affected_hypotheses reference. Every recommendation MUST "
        "include at least one supporting_evidence reference. "
        "Be specific, not vague."
    )

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str = "gpt-4o-mini",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the insights generator with optional injected
        dependencies.

        Args:
            client: ``AsyncOpenAI`` instance. A default is created if
                not supplied.
            model: OpenAI chat model (default ``gpt-4o-mini``).
            logger: Logger instance. Defaults to
                ``logging.getLogger(__name__)``.
        """
        self._client = client or AsyncOpenAI()
        self._model = model
        self._logger = logger or logging.getLogger(__name__)

    async def generate(
        self,
        hypotheses: list[HypothesisSummary],
        program_contexts: list[ProgramContext] | None = None,
    ) -> InsightsReport | None:
        """Generate cross-hypothesis insights via LLM.

        Builds a user prompt from serialized hypotheses and optional
        program contexts, makes a single ``chat.completions.create()``
        call with ``response_format="json_object"``, and parses the
        result into an ``InsightsReport``.

        Returns ``InsightsReport`` on success, ``None`` on any failure
        (API error, JSON parse failure, malformed response). Partial
        responses (missing sections) are filled with empty lists.
        """
        hypotheses_data = [asdict(h) for h in hypotheses]
        contexts_data = [asdict(ctx) for ctx in (program_contexts or [])]

        user_prompt = (
            f"Experiment Results:\n"
            f"{json.dumps(hypotheses_data, indent=2)}\n\n"
            f"Program Contexts (per-hypothesis baseline, direction, "
            f"success criteria):\n"
            f"{json.dumps(contexts_data, indent=2)}"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.INSIGHTS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            self._logger.warning("LLM insights generation failed: %s", exc)
            return None

        content = response.choices[0].message.content
        if content is None:
            self._logger.warning(
                "LLM returned empty response for insights generation"
            )
            return None

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            self._logger.warning(
                "Failed to parse insights JSON: %s", content[:200],
            )
            return None

        if not isinstance(result, dict):
            self._logger.warning(
                "Insights response is not a dict: %s", content[:200],
            )
            return None

        # --- Partial parse resilience: missing sections → empty lists ---
        patterns_data = result.get("patterns", [])
        recommendations_data = result.get("recommendations", [])
        evidence_data = result.get("evidence_correlation", [])

        if not isinstance(patterns_data, list):
            patterns_data = []
        if not isinstance(recommendations_data, list):
            recommendations_data = []
        if not isinstance(evidence_data, list):
            evidence_data = []

        # --- Parse individual items; skip malformed dicts ---
        patterns: list[PatternInsight] = []
        for item in patterns_data:
            if isinstance(item, dict):
                try:
                    patterns.append(
                        PatternInsight(
                            title=str(item.get("title", "")),
                            description=str(item.get("description", "")),
                            affected_hypotheses=list(
                                item.get("affected_hypotheses", [])
                            ),
                            confidence=str(
                                item.get("confidence", "medium")
                            ),
                        )
                    )
                except Exception:
                    pass

        recommendations: list[Recommendation] = []
        for item in recommendations_data:
            if isinstance(item, dict):
                try:
                    recommendations.append(
                        Recommendation(
                            action=str(item.get("action", "")),
                            priority=str(item.get("priority", "medium")),
                            rationale=str(item.get("rationale", "")),
                            supporting_evidence=list(
                                item.get("supporting_evidence", [])
                            ),
                        )
                    )
                except Exception:
                    pass

        evidence_correlations: list[EvidenceCorrelation] = []
        for item in evidence_data:
            if isinstance(item, dict):
                try:
                    evidence_correlations.append(
                        EvidenceCorrelation(
                            paper_id=str(item.get("paper_id", "")),
                            evidence_summary=str(
                                item.get("evidence_summary", "")
                            ),
                            related_hypotheses=list(
                                item.get("related_hypotheses", [])
                            ),
                            code_areas=list(item.get("code_areas", [])),
                        )
                    )
                except Exception:
                    pass

        return InsightsReport(
            patterns=patterns,
            recommendations=recommendations,
            evidence_correlation=evidence_correlations,
        )
