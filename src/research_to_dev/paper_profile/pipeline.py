"""Profiling Pipeline — single-stage orchestrator for paper profiling.

Orchestrates the paper profiling stage sequentially for each
``RankedPaper``. Follows the same constructor-injection pattern
as ``RankingPipeline``.
"""

from __future__ import annotations

import logging

from research_to_dev.paper_profile.protocols import SectionExtractor
from research_to_dev.paper_profile.stages import profile_paper
from research_to_dev.paper_profile.types import (
    PaperProfile,
    ProfilingError,
    ProfilingResult,
)
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import ProfileConfig


class ProfilingPipeline:
    """Single-stage pipeline for extracting sections and claims from papers.

    Constructor-injected config and protocol implementation keep the
    pipeline testable without real API calls. The stage is error-resilient
    (RF-05): any failure produces a degraded ``PaperProfile`` so the
    pipeline never raises.
    """

    def __init__(
        self,
        extractor: SectionExtractor,
        *,
        config: ProfileConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._extractor = extractor
        self._config = config or ProfileConfig()
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def profile(
        self, papers: list[RankedPaper]
    ) -> ProfilingResult:
        """Run profiling on all ranked papers.

        Processes papers sequentially (AD-08). On failure, a degraded
        ``PaperProfile`` (empty sections and claims) is still returned,
        and the error is recorded in ``ProfilingResult.errors``.

        Args:
            papers: Ranked papers to profile.

        Returns:
            ``ProfilingResult`` with one ``PaperProfile`` per input
            paper and any ``ProfilingError`` entries.
        """
        total_input = len(papers)

        # Empty input short-circuit (Spec: Empty input)
        if total_input == 0:
            return ProfilingResult(profiles=[], total_input=0)

        profiles: list[PaperProfile] = []
        errors: list[ProfilingError] = []

        for rp in papers:
            try:
                profile, error = await profile_paper(
                    rp,
                    extractor=self._extractor,
                    config=self._config,
                )
                profiles.append(profile)
                if error is not None:
                    errors.append(error)
            except Exception as exc:
                # Ultimate safety net — should never reach here because
                # profile_paper() already catches everything, but belt
                # and suspenders.
                self._logger.error(
                    "Unexpected error profiling paper '%s': %s",
                    rp.paper.title,
                    exc,
                )
                profiles.append(PaperProfile(ranked_paper=rp))
                errors.append(
                    ProfilingError(
                        paper_title=rp.paper.title,
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                )

        return ProfilingResult(
            profiles=profiles,
            total_input=total_input,
            errors=errors,
        )
