"""Pipeline orchestrator — wires all 7 modules into a single ``run()`` call.

Replicates the E2E script's wiring pattern with constructor injection and
a progress callback for CLI integration.

Data contracts:
    - ``StepTrace`` — per-step status, error, and warnings
    - ``PipelineTrace`` — full pipeline output with all 7 module results,
      serialisable via ``to_json()``
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal

from openai import AsyncOpenAI

from research_to_dev.codebase.adapters import OpenAIDescriber
from research_to_dev.codebase.pipeline import CodebaseAnalysisPipeline
from research_to_dev.correlation.adapters import OpenAICorrelationClassifier
from research_to_dev.correlation.pipeline import CorrelationPipeline
from research_to_dev.extraction.pipeline import ExtractionPipeline
from research_to_dev.extraction.scraper import HttpxScraper
from research_to_dev.extraction.types import (
    AcademicContent,
    ExtractionResult,
    GeneralContent,
)
from research_to_dev.hypothesis.adapters import OpenAIHypothesisGenerator
from research_to_dev.hypothesis.pipeline import HypothesisPipeline
from research_to_dev.paper_profile.adapters import OpenAISectionExtractor
from research_to_dev.paper_profile.pipeline import ProfilingPipeline
from research_to_dev.ranking.adapters import OpenAIEmbedder, OpenAIJudge
from research_to_dev.ranking.pipeline import RankingPipeline
from research_to_dev.ranking.types import RankingResult
from research_to_dev.retriever import (
    ArxivRetriever,
    RetrieverOrchestrator,
    SemanticScholarRetriever,
    TavilyRetriever,
)
from research_to_dev.retriever.protocol import RetrievalResult
from research_to_dev.codebase.types import CodebaseContext
from research_to_dev.correlation.types import CorrelationMap
from research_to_dev.hypothesis.types import HypothesisResult
from research_to_dev.paper_profile.types import ProfilingResult
from research_to_dev.shared.config import (
    CodebaseConfig,
    CorrelationMapConfig,
    HypothesisConfig,
    ProfileConfig,
    RankingConfig,
)

# ======================================================================
# Data contracts
# ======================================================================


@dataclass
class StepTrace:
    """Per-step execution status captured during the pipeline run.

    ``status`` values:
        - ``"success"`` — step completed normally
        - ``"error"`` — step raised an exception
        - ``"skipped"`` — step was skipped because its input was empty
        - ``"empty"`` — step completed but produced zero results
    """

    status: Literal["success", "error", "skipped", "empty"]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineTrace:
    """Full output of one pipeline execution — all 7 steps, all results.

    Serialisable via ``to_json()`` which uses ``dataclasses.asdict`` +
    ``json.dumps(indent=2)``.
    """

    query: str
    codebase_path: str
    timestamp: str  # ISO 8601
    steps: dict[str, StepTrace] = field(default_factory=dict)

    retrieval: RetrievalResult | None = None
    extraction: ExtractionResult | None = None
    ranking: RankingResult | None = None
    profiling: ProfilingResult | None = None
    codebase: CodebaseContext | None = None
    correlation: CorrelationMap | None = None
    hypothesis: HypothesisResult | None = None

    hypotheses: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full trace to a JSON string."""
        raw = asdict(self)
        return json.dumps(raw, indent=indent, ensure_ascii=False)


# ======================================================================
# Orchestrator
# ======================================================================


class PipelineOrchestrator:
    """Wires all 7 pipeline modules into a single ``run()`` call.

    Accepts a single shared ``AsyncOpenAI`` client and an optional
    ``on_step`` progress callback.  Every step is wrapped in try/except;
    empty results trigger fail-fast; partial failures continue with
    warnings collected in the trace.

    Usage::

        client = AsyncOpenAI()
        orchestrator = PipelineOrchestrator(client, on_step=callback)
        trace = await orchestrator.run("my query", "./src")
        print(trace.to_json())
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        on_step: Callable[[str, int], None] | None = None,
    ) -> None:
        self._client = client
        self._on_step = on_step

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, query: str, codebase_path: str) -> PipelineTrace:
        """Execute all 7 pipeline steps sequentially.

        Returns a ``PipelineTrace`` even if some steps fail — the caller
        inspects ``trace.steps`` for per-step status.
        """
        t0 = time.monotonic()

        trace = PipelineTrace(
            query=query,
            codebase_path=codebase_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # -- shared adapters (reused across multiple steps) ------------
        embedder = OpenAIEmbedder(client=self._client)

        # -- step 1: retrieval ----------------------------------------
        self._emit("Searching papers across sources", 1)

        retrievers: list = [ArxivRetriever(), SemanticScholarRetriever()]
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            retrievers.append(TavilyRetriever(api_key=tavily_key))

        orchestrator = RetrieverOrchestrator(retrievers=retrievers)

        retrieval_result: RetrievalResult | None = None
        try:
            retrieval_result = await orchestrator.search(query, max_results=10)
            trace.retrieval = retrieval_result
        except Exception as exc:
            msg = str(exc)
            trace.steps["retrieval"] = StepTrace(status="error", error=msg)
            trace.warnings.append(f"retrieval: {msg}")
            self._emit(f"❌ Retrieval failed: {msg}", 1)
            return trace

        if not retrieval_result.results:
            trace.steps["retrieval"] = StepTrace(status="empty")
            self._emit("No papers found — pipeline stopped", 1)
            return trace

        trace.steps["retrieval"] = StepTrace(
            status="success",
            warnings=(
                [f"[{e.source}] {e.message}" for e in retrieval_result.errors]
                if retrieval_result.errors
                else []
            ),
        )
        self._emit(
            f"Found {len(retrieval_result.results)} papers",
            1,
        )

        # -- step 2: extraction ---------------------------------------
        self._emit("Extracting content from papers", 2)

        scraper = HttpxScraper()
        extraction_pipeline = ExtractionPipeline(scraper=scraper)

        extraction_result: ExtractionResult | None = None
        academic_contents: list[AcademicContent] = []
        general_contents: list[GeneralContent] = []
        try:
            extraction_result = await extraction_pipeline.extract(retrieval_result)
            trace.extraction = extraction_result
            academic_contents = [
                c
                for c in extraction_result.contents
                if isinstance(c, AcademicContent)
            ]
            general_contents = [
                c
                for c in extraction_result.contents
                if isinstance(c, GeneralContent)
            ]
        except Exception as exc:
            msg = str(exc)
            trace.steps["extraction"] = StepTrace(status="error", error=msg)
            trace.warnings.append(f"extraction: {msg}")
            self._emit(f"⚠️ Extraction failed: {msg}", 2)
            # Continue — extraction failure doesn't prevent downstream steps
            # (they will skip because academic_contents is empty)

        if extraction_result and not extraction_result.contents:
            trace.steps["extraction"] = StepTrace(status="empty")
            self._emit("No content extracted — pipeline stopped", 2)
            return trace

        if extraction_result:
            trace.steps["extraction"] = StepTrace(
                status="success",
                warnings=(
                    [f"[{e.source}] {e.message}" for e in extraction_result.errors]
                    if extraction_result.errors
                    else []
                ),
            )
            self._emit(
                f"Extracted {len(academic_contents)} academic + "
                f"{len(general_contents)} general items",
                2,
            )

        # -- step 3: ranking ------------------------------------------
        self._emit("Ranking papers by relevance", 3)

        judge = OpenAIJudge(client=self._client)
        ranking_config = RankingConfig()
        ranking_pipeline = RankingPipeline(
            embedder=embedder, judge=judge, config=ranking_config
        )

        ranking_result: RankingResult | None = None
        ranked_papers: list = []
        if not academic_contents:
            ranking_result = RankingResult(
                papers=[], total_input=0, filtered_by_metadata=0, semantically_ranked=0
            )
            trace.ranking = ranking_result
            trace.steps["ranking"] = StepTrace(status="skipped")
            self._emit("No academic content — ranking skipped", 3)
        else:
            try:
                ranking_result = await ranking_pipeline.rank(academic_contents, query)
                trace.ranking = ranking_result
                ranked_papers = ranking_result.papers
            except Exception as exc:
                msg = str(exc)
                trace.steps["ranking"] = StepTrace(status="error", error=msg)
                trace.warnings.append(f"ranking: {msg}")
                self._emit(f"⚠️ Ranking failed: {msg}", 3)

            if ranking_result and not ranking_result.papers:
                trace.steps["ranking"] = StepTrace(status="empty")
                self._emit("No papers ranked — pipeline stopped", 3)
                return trace

            if ranking_result:
                trace.steps["ranking"] = StepTrace(status="success")
                self._emit(
                    f"Ranked: {len(ranking_result.papers)} papers selected", 3
                )

        # -- step 4: profiling ----------------------------------------
        self._emit("Profiling papers (sections + claims)", 4)

        section_extractor = OpenAISectionExtractor(client=self._client)
        profile_config = ProfileConfig()
        profiling_pipeline = ProfilingPipeline(
            extractor=section_extractor, config=profile_config
        )

        profiling_result: object | None = None
        profiles: list = []
        if not ranked_papers:
            from research_to_dev.paper_profile.types import ProfilingResult as PR

            profiling_result = PR(profiles=[], total_input=0)
            trace.profiling = profiling_result
            trace.steps["profiling"] = StepTrace(status="skipped")
            self._emit("No ranked papers — profiling skipped", 4)
        else:
            try:
                profiling_result = await profiling_pipeline.profile(ranked_papers)
                trace.profiling = profiling_result
                profiles = profiling_result.profiles
            except Exception as exc:
                msg = str(exc)
                trace.steps["profiling"] = StepTrace(status="error", error=msg)
                trace.warnings.append(f"profiling: {msg}")
                self._emit(f"⚠️ Profiling failed: {msg}", 4)

            if profiling_result and not getattr(profiling_result, "profiles", None):
                trace.steps["profiling"] = StepTrace(status="empty")
                self._emit("No profiles generated — pipeline stopped", 4)
                return trace

            if profiling_result:
                trace.steps["profiling"] = StepTrace(
                    status="success",
                    warnings=(
                        [
                            f"[{e.paper_title}] {e.message}"
                            for e in profiling_result.errors
                        ]
                        if profiling_result.errors
                        else []
                    ),
                )
                n_claims = sum(len(p.claims) for p in profiles)
                self._emit(
                    f"Profiled {len(profiles)} papers ({n_claims} claims)", 4
                )

        # -- step 5: codebase analysis --------------------------------
        self._emit("Analyzing codebase", 5)

        describer = OpenAIDescriber(client=self._client)
        codebase_config = CodebaseConfig()
        codebase_pipeline = CodebaseAnalysisPipeline(
            describer=describer, config=codebase_config
        )

        codebase_result: object | None = None
        try:
            codebase_result = await codebase_pipeline.analyze(codebase_path)
            trace.codebase = codebase_result
        except Exception as exc:
            msg = str(exc)
            trace.steps["codebase"] = StepTrace(status="error", error=msg)
            trace.warnings.append(f"codebase: {msg}")
            self._emit(f"⚠️ Codebase analysis failed: {msg}", 5)

        if codebase_result:
            trace.steps["codebase"] = StepTrace(status="success")
            self._emit(
                f"Codebase: {codebase_result.total_files_scanned} files, "
                f"{len(codebase_result.components)} components, "
                f"{len(codebase_result.modules)} modules",
                5,
            )

        # -- step 6: correlation --------------------------------------
        self._emit("Correlating claims with codebase", 6)

        corr_classifier = OpenAICorrelationClassifier(client=self._client)
        corr_config = CorrelationMapConfig()
        correlation_pipeline = CorrelationPipeline(
            embedder=embedder, classifier=corr_classifier, config=corr_config
        )

        correlation_result: object | None = None
        if not profiles:
            from research_to_dev.correlation.types import CorrelationMap

            correlation_result = CorrelationMap(
                papers_analyzed=0,
                total_claims=0,
                total_components=(
                    len(codebase_result.components) if codebase_result else 0
                ),
                total_modules=(
                    len(codebase_result.modules) if codebase_result else 0
                ),
            )
            trace.correlation = correlation_result
            trace.steps["correlation"] = StepTrace(status="skipped")
            self._emit("No profiles — correlation skipped", 6)
        elif not codebase_result:
            from research_to_dev.correlation.types import CorrelationMap

            correlation_result = CorrelationMap(
                papers_analyzed=len(profiles) if profiles else 0,
                total_claims=sum(len(p.claims) for p in profiles) if profiles else 0,
            )
            trace.correlation = correlation_result
            trace.steps["correlation"] = StepTrace(status="skipped")
            self._emit("No codebase context — correlation skipped", 6)
        else:
            try:
                correlation_result = await correlation_pipeline.correlate(
                    profiles=profiles,
                    codebase=codebase_result,
                )
                trace.correlation = correlation_result
            except Exception as exc:
                msg = str(exc)
                trace.steps["correlation"] = StepTrace(status="error", error=msg)
                trace.warnings.append(f"correlation: {msg}")
                self._emit(f"⚠️ Correlation failed: {msg}", 6)

            if correlation_result:
                trace.steps["correlation"] = StepTrace(status="success")
                self._emit(
                    f"Correlation: {correlation_result.matches_found} matches "
                    f"({len(correlation_result.correlations)} component + "
                    f"{len(correlation_result.module_correlations)} module)",
                    6,
                )

        # -- step 7: hypothesis generation ----------------------------
        self._emit("Generating hypotheses", 7)

        hypo_generator = OpenAIHypothesisGenerator(client=self._client)
        hypo_config = HypothesisConfig()
        hypothesis_pipeline = HypothesisPipeline(
            generator=hypo_generator, embedder=embedder, config=hypo_config
        )

        hypothesis_result: object | None = None
        if not correlation_result:
            from research_to_dev.hypothesis.types import HypothesisResult as HR

            hypothesis_result = HR()
            trace.hypothesis = hypothesis_result
            trace.steps["hypothesis"] = StepTrace(status="skipped")
            self._emit("No correlation map — hypothesis skipped", 7)
        else:
            try:
                hypothesis_result = await hypothesis_pipeline.run(
                    correlation_map=correlation_result,
                    query=query,
                    general_content=general_contents,
                )
                trace.hypothesis = hypothesis_result
            except Exception as exc:
                msg = str(exc)
                trace.steps["hypothesis"] = StepTrace(status="error", error=msg)
                trace.warnings.append(f"hypothesis: {msg}")
                self._emit(f"⚠️ Hypothesis failed: {msg}", 7)

            if hypothesis_result:
                trace.steps["hypothesis"] = StepTrace(status="success")
                self._emit(
                    f"Hypothesis: {len(hypothesis_result.hypotheses)} hypotheses, "
                    f"{hypothesis_result.reflection_rounds_completed} reflection rounds",
                    7,
                )

                if hypothesis_result.errors:
                    for err in hypothesis_result.errors[:3]:
                        trace.warnings.append(f"hypothesis: {err.message}")

        # -- post-process: prepare hypotheses list for readability ----
        if hypothesis_result and hypothesis_result.hypotheses:
            sorted_h = sorted(
                hypothesis_result.hypotheses,
                key=lambda h: h.composite,
                reverse=True,
            )
            trace.hypotheses = [
                {
                    "id": h.id,
                    "title": h.title,
                    "composite": h.composite,
                    "description": h.description,
                    "supporting_papers": h.supporting_papers,
                    "approach": h.approach,
                    "target_metric": h.target_metric,
                    "expected_improvement": h.expected_improvement,
                    "code_changes": h.code_changes,
                    "success_criteria": h.success_criteria,
                    "scores": {
                        "relevance": h.scores.relevance,
                        "feasibility": h.scores.feasibility,
                        "evidence": h.scores.evidence,
                    },
                    "correlations": h.correlations,
                }
                for h in sorted_h
            ]

        # -- collect all step warnings into trace-level warnings -------
        for step_trace in trace.steps.values():
            trace.warnings.extend(step_trace.warnings)

        trace.warnings = list(dict.fromkeys(  # deduplicate, preserve order
            trace.warnings
        ))

        _elapsed = time.monotonic() - t0
        self._emit(f"Pipeline complete ({_elapsed:.1f}s)", 0)

        return trace

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, msg: str, step: int) -> None:
        """Call the progress callback if one was provided."""
        if self._on_step is not None:
            self._on_step(msg, step)
