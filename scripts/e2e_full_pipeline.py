#!/usr/bin/env python3
"""End-to-end test: full 7-step research-to-code pipeline with real API calls.

Wires all 7 modules in sequence:
    1. Retriever      — multi-source paper search (arXiv, Semantic Scholar, Tavily)
    2. Extraction     — content fetching + typing (academic vs general)
    3. Ranking        — 3-stage funnel (metadata → semantic → LLM judge)
    4. Profiling      — section & claim extraction per paper
    5. Codebase       — AST scan + LLM component descriptions (self-analyze src/)
    6. Correlation    — embed → cosine → filter → classify → assemble
    7. Hypothesis     — generate → reflect → score → ground → rank

Requires: OPENAI_API_KEY set in a ``.env`` file at the project root.
Optional: TAVILY_API_KEY — if missing, Tavily is skipped gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure the package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Load API keys from .env (manual parsing — no extra deps needed)
# ---------------------------------------------------------------------------
def _load_api_key(key_name: str) -> str | None:
    """Parse a key from the project-root ``.env`` file.

    Returns ``None`` if the file or key is missing — callers decide
    whether that's fatal or graceful.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key_name:
            val = v.strip().strip('"').strip("'")
            return val if val else None

    return None


OPENAI_API_KEY = _load_api_key("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not found in .env")
    sys.exit(1)

TAVILY_API_KEY = _load_api_key("TAVILY_API_KEY")
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
if TAVILY_API_KEY:
    os.environ.setdefault("TAVILY_API_KEY", TAVILY_API_KEY)

# ---------------------------------------------------------------------------
# Now import project modules (API key must be in env before AsyncOpenAI init)
# ---------------------------------------------------------------------------
from openai import AsyncOpenAI

from research_to_dev.codebase.adapters import OpenAIDescriber
from research_to_dev.codebase.pipeline import CodebaseAnalysisPipeline
from research_to_dev.codebase.types import CodebaseContext
from research_to_dev.correlation.adapters import OpenAICorrelationClassifier
from research_to_dev.correlation.pipeline import CorrelationPipeline
from research_to_dev.correlation.types import CorrelationMap
from research_to_dev.extraction.pipeline import ExtractionPipeline
from research_to_dev.extraction.scraper import HttpxScraper
from research_to_dev.extraction.types import (
    AcademicContent,
    ExtractionResult,
    GeneralContent,
)
from research_to_dev.hypothesis.adapters import OpenAIHypothesisGenerator
from research_to_dev.hypothesis.pipeline import HypothesisPipeline
from research_to_dev.hypothesis.types import HypothesisResult
from research_to_dev.paper_profile.adapters import OpenAISectionExtractor
from research_to_dev.paper_profile.pipeline import ProfilingPipeline
from research_to_dev.paper_profile.types import PaperProfile, ProfilingResult
from research_to_dev.ranking.adapters import OpenAIEmbedder, OpenAIJudge
from research_to_dev.ranking.pipeline import RankingPipeline
from research_to_dev.ranking.types import RankedPaper, RankingResult
from research_to_dev.retriever.arxiv import ArxivRetriever
from research_to_dev.retriever.orchestrator import RetrieverOrchestrator
from research_to_dev.retriever.protocol import RetrievalResult
from research_to_dev.retriever.semantic_scholar import SemanticScholarRetriever
from research_to_dev.retriever.tavily import TavilyRetriever
from research_to_dev.shared.config import (
    CodebaseConfig,
    CorrelationMapConfig,
    HypothesisConfig,
    ProfileConfig,
    RankingConfig,
)


# ======================================================================
# Constants
# ======================================================================

QUERY = (
    "How to optimize the query input for precise queries "
    "vs open-ended research questions"
)

CODEBASE_PATH = str(SRC.resolve())

# ======================================================================
# Report formatting
# ======================================================================


def print_header(text: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {text}")
    print(f"{'=' * 72}")


def print_subheader(text: str) -> None:
    print(f"\n{'-' * 72}")
    print(f"  {text}")
    print(f"{'-' * 72}")


def truncate(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


# ======================================================================
# Step result containers (for error-resilient report)
# ======================================================================

from dataclasses import dataclass, field


@dataclass
class StepResults:
    """Hold results and errors from each pipeline step."""

    retrieval: RetrievalResult | None = None
    retrieval_error: str | None = None

    extraction: ExtractionResult | None = None
    extraction_error: str | None = None
    academic_contents: list[AcademicContent] = field(default_factory=list)
    general_contents: list[GeneralContent] = field(default_factory=list)

    ranking: RankingResult | None = None
    ranking_error: str | None = None
    ranked_papers: list[RankedPaper] = field(default_factory=list)

    profiling: ProfilingResult | None = None
    profiling_error: str | None = None
    profiles: list[PaperProfile] = field(default_factory=list)

    codebase: CodebaseContext | None = None
    codebase_error: str | None = None

    correlation: CorrelationMap | None = None
    correlation_error: str | None = None

    hypothesis: HypothesisResult | None = None
    hypothesis_error: str | None = None


def print_full_report(results: StepResults, elapsed: float) -> None:
    """Print a comprehensive report of all 7 pipeline steps."""

    print_header("E2E FULL PIPELINE — REPORT")
    print(f"\n  Query:          \"{QUERY}\"")
    print(f"  Codebase path:  {CODEBASE_PATH}")
    print(f"  Total time:     {elapsed:.1f}s")

    # --- Step 1: Retriever ---
    print_subheader("STEP 1: RETRIEVER — Multi-Source Search")
    if results.retrieval:
        r = results.retrieval
        print(f"  Total results:   {len(r.results)}")
        print(f"  Errors:          {len(r.errors)}")
        for err in r.errors:
            print(f"    - [{err.source}] {err.message}")
        # Source breakdown
        sources: dict[str, int] = {}
        for sr in r.results:
            sources[sr.source] = sources.get(sr.source, 0) + 1
        print(f"  By source:")
        for src, count in sorted(sources.items()):
            print(f"    - {src}: {count}")
        if TAVILY_API_KEY is None:
            print(f"  ⚠️  Tavily skipped — TAVILY_API_KEY not set")
    elif results.retrieval_error:
        print(f"  ❌ FAILED: {results.retrieval_error}")

    # --- Step 2: Extraction ---
    print_subheader("STEP 2: EXTRACTION — Content Fetching + Typing")
    if results.extraction:
        print(f"  Academic items:  {len(results.academic_contents)}")
        print(f"  General items:   {len(results.general_contents)}")
        print(f"  Errors:          {len(results.extraction.errors)}")
        for err in results.extraction.errors[:5]:
            print(f"    - [{err.source}] {truncate(err.message, 60)}")
        if len(results.extraction.errors) > 5:
            print(f"    ... and {len(results.extraction.errors) - 5} more")
    elif results.extraction_error:
        print(f"  ❌ FAILED: {results.extraction_error}")

    # --- Step 3: Ranking ---
    print_subheader("STEP 3: RANKING — 3-Stage Funnel")
    if results.ranking:
        rr = results.ranking
        print(f"  Total input:           {rr.total_input}")
        print(f"  Filtered by metadata:  {rr.filtered_by_metadata}")
        print(f"  Semantically ranked:   {rr.semantically_ranked}")
        print(f"  Final papers:          {len(rr.papers)}")
        if rr.papers:
            for i, rp in enumerate(rr.papers[:5], 1):
                print(f"    [{i}] {truncate(rp.paper.title, 50)} "
                      f"(rel={rp.relevance_score:.2f}, sem={rp.semantic_score:.2f})")
            if len(rr.papers) > 5:
                print(f"    ... and {len(rr.papers) - 5} more")
    elif results.ranking_error:
        print(f"  ❌ FAILED: {results.ranking_error}")

    # --- Step 4: Profiling ---
    print_subheader("STEP 4: PROFILING — Section & Claim Extraction")
    if results.profiling:
        pr = results.profiling
        print(f"  Total input:   {pr.total_input}")
        print(f"  Profiles:      {len(pr.profiles)}")
        total_claims = sum(len(p.claims) for p in pr.profiles)
        print(f"  Total claims:  {total_claims}")
        print(f"  Errors:        {len(pr.errors)}")
        for err in pr.errors[:3]:
            print(f"    - {truncate(err.paper_title, 50)}: {truncate(err.message, 60)}")
    elif results.profiling_error:
        print(f"  ❌ FAILED: {results.profiling_error}")

    # --- Step 5: Codebase ---
    print_subheader("STEP 5: CODEBASE — Self-Analyze src/")
    if results.codebase:
        cb = results.codebase
        print(f"  Project:           {cb.project_name}")
        print(f"  Language:          {cb.language}")
        print(f"  Files scanned:     {cb.total_files_scanned}")
        print(f"  Components found:  {cb.total_components_found}")
        print(f"  Components:        {len(cb.components)}")
        print(f"  Modules:           {len(cb.modules)}")
        if cb.modules:
            for m in cb.modules[:5]:
                print(f"    - {m.module_name}: {truncate(m.summary, 60)}")
            if len(cb.modules) > 5:
                print(f"    ... and {len(cb.modules) - 5} more")
    elif results.codebase_error:
        print(f"  ❌ FAILED: {results.codebase_error}")

    # --- Step 6: Correlation ---
    print_subheader("STEP 6: CORRELATION — Embed → Cosine → Classify → Assemble")
    if results.correlation:
        cm = results.correlation
        print(f"  Papers analyzed:      {cm.papers_analyzed}")
        print(f"  Total claims:         {cm.total_claims}")
        print(f"  Total components:     {cm.total_components}")
        print(f"  Total modules:        {cm.total_modules}")
        print(f"  Matches found:        {cm.matches_found}")
        print(f"    - Component corrs:  {len(cm.correlations)}")
        print(f"    - Module corrs:     {len(cm.module_correlations)}")
        if cm.correlations:
            for i, c in enumerate(cm.correlations[:5], 1):
                print(f"    [{i}] {c.correlation_type} | "
                      f"{truncate(c.paper_title, 40)} → {c.component.name!r} "
                      f"(score={c.similarity_score:.4f})")
            if len(cm.correlations) > 5:
                print(f"    ... and {len(cm.correlations) - 5} more")
        if cm.module_correlations:
            for i, mc in enumerate(cm.module_correlations[:3], 1):
                print(f"    [M{i}] {mc.correlation_type} | "
                      f"{truncate(mc.paper_title, 40)} → {mc.module.module_name!r} "
                      f"(score={mc.similarity_score:.4f})")
    elif results.correlation_error:
        print(f"  ❌ FAILED: {results.correlation_error}")

    # --- Step 7: Hypothesis ---
    print_subheader("STEP 7: HYPOTHESIS — Generate → Reflect → Score → Ground → Rank")
    if results.hypothesis:
        hr = results.hypothesis
        print(f"  Total input papers:     {hr.total_input_papers}")
        print(f"  Total correlations:     {hr.total_correlations}")
        print(f"  Reflection rounds:      {hr.reflection_rounds_completed}")
        print(f"  Converged:              {hr.converged}")
        print(f"  Total hypotheses:       {len(hr.hypotheses)}")
        print(f"  Errors:                 {len(hr.errors)}")

        if hr.errors:
            for err in hr.errors[:5]:
                print(f"    - {truncate(err.message, 100)}")
            if len(hr.errors) > 5:
                print(f"    ... and {len(hr.errors) - 5} more")

        if hr.hypotheses:
            ranked = sorted(hr.hypotheses, key=lambda h: h.composite, reverse=True)
            print(f"\n  Top hypotheses (by composite score):")
            for i, h in enumerate(ranked[:5], 1):
                print(f"  [{i}] \"{truncate(h.title, 50)}\"")
                print(f"      Desc:     {truncate(h.description, 100)}")
                print(f"      Scores:   rel={h.scores.relevance}  "
                      f"feas={h.scores.feasibility}  evid={h.scores.evidence}")
                print(f"      Composite: {h.composite:.2f}")
                if h.supporting_papers:
                    papers_str = ", ".join(truncate(p, 35) for p in h.supporting_papers[:2])
                    print(f"      Papers:   {papers_str}")
            if len(ranked) > 5:
                print(f"    ... and {len(ranked) - 5} more")

            # Score distribution
            score_dist: dict[str, int] = {}
            for h in ranked:
                bucket = str(int(h.composite))
                score_dist[bucket] = score_dist.get(bucket, 0) + 1
            print(f"\n  Composite score distribution:")
            for bucket in sorted(score_dist, key=int, reverse=True):
                count = score_dist[bucket]
                bar = "#" * count
                print(f"    {bucket:>2}: {bar} ({count})")

            avg_comp = sum(h.composite for h in ranked) / len(ranked)
            print(f"\n  Average composite:   {avg_comp:.2f}")
            print(f"  Best composite:      {ranked[0].composite:.2f}")
            print(f"  Worst composite:     {ranked[-1].composite:.2f}")
    elif results.hypothesis_error:
        print(f"  ❌ FAILED: {results.hypothesis_error}")

    # --- Final summary ---
    print_header("RESULT")
    steps_ok = sum([
        1 if results.retrieval else 0,
        1 if results.extraction else 0,
        1 if results.ranking else 0,
        1 if results.profiling else 0,
        1 if results.codebase else 0,
        1 if results.correlation else 0,
        1 if results.hypothesis else 0,
    ])
    steps_failed = 7 - steps_ok

    if steps_failed == 0:
        print(f"\n  ✅ SUCCESS: All 7 pipeline steps completed.")
        if results.hypothesis and results.hypothesis.hypotheses:
            print(f"  Generated {len(results.hypothesis.hypotheses)} grounded hypotheses.")
        elif results.hypothesis:
            print(f"  Pipeline ran but produced 0 hypotheses "
                  f"(no correlations to generate from).")
    else:
        print(f"\n  ⚠️  PARTIAL: {steps_ok}/7 steps succeeded, {steps_failed} failed.")
        print(f"  Check the per-step output above for error details.")

    print(f"\n  Total wall time: {elapsed:.1f}s")
    print(f"{'=' * 72}\n")


# ======================================================================
# Main
# ======================================================================


async def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import time
    t0 = time.monotonic()

    results = StepResults()

    print_header("E2E FULL PIPELINE — Real API Calls Across All 7 Modules")
    print(f"\n  Query:          \"{QUERY}\"")
    print(f"  Codebase path:  {CODEBASE_PATH}")
    if TAVILY_API_KEY:
        print(f"  Tavily key:     loaded ✅")
    else:
        print(f"  Tavily key:     not set — will skip Tavily retriever")

    # ------------------------------------------------------------------
    # Create single AsyncOpenAI client — shared across all adapters
    # ------------------------------------------------------------------
    print("\n  ⏳ Creating shared OpenAI client...")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print("  ✅ Client ready")

    # ------------------------------------------------------------------
    # STEP 1: Retriever
    # ------------------------------------------------------------------
    print_header("STEP 1: RETRIEVER — Searching 3 sources in parallel")
    print("  ⏳ Querying arXiv, Semantic Scholar, Tavily...")

    retrievers = [
        ArxivRetriever(),
        SemanticScholarRetriever(),
    ]
    if TAVILY_API_KEY:
        retrievers.append(TavilyRetriever(api_key=TAVILY_API_KEY))
    else:
        print("  ⚠️  Skipping Tavily — TAVILY_API_KEY not found in .env")

    orchestrator = RetrieverOrchestrator(retrievers=retrievers)

    try:
        retrieval_result = await orchestrator.search(QUERY, max_results=10)
        results.retrieval = retrieval_result
        print(f"  ✅ Found {len(retrieval_result.results)} results "
              f"({len(retrieval_result.errors)} source errors)")
        if retrieval_result.errors:
            for err in retrieval_result.errors:
                print(f"    ⚠️  [{err.source}] {err.message}")
    except Exception as exc:
        results.retrieval_error = str(exc)
        print(f"  ❌ Retriever failed: {exc}")
        print_full_report(results, time.monotonic() - t0)
        return

    if not retrieval_result.results:
        print("  ⚠️  No results found — pipeline cannot continue.")
        print_full_report(results, time.monotonic() - t0)
        return

    # ------------------------------------------------------------------
    # STEP 2: Extraction
    # ------------------------------------------------------------------
    print_header("STEP 2: EXTRACTION — Fetching content from results")
    print("  ⏳ Scraping papers and articles...")

    scraper = HttpxScraper()
    extraction_pipeline = ExtractionPipeline(scraper=scraper)

    try:
        extraction_result = await extraction_pipeline.extract(retrieval_result)
        results.extraction = extraction_result
        results.academic_contents = [
            c for c in extraction_result.contents if isinstance(c, AcademicContent)
        ]
        results.general_contents = [
            c for c in extraction_result.contents if isinstance(c, GeneralContent)
        ]
        print(f"  ✅ Extracted {len(results.academic_contents)} academic + "
              f"{len(results.general_contents)} general items "
              f"({len(extraction_result.errors)} errors)")
    except Exception as exc:
        results.extraction_error = str(exc)
        print(f"  ❌ Extraction failed: {exc}")

    # ------------------------------------------------------------------
    # STEP 3: Ranking
    # ------------------------------------------------------------------
    print_header("STEP 3: RANKING — 3-Stage Funnel (metadata → semantic → LLM)")
    print("  ⏳ Creating embedder + judge + pipeline...")

    embedder = OpenAIEmbedder(client=client)
    judge = OpenAIJudge(client=client)
    ranking_config = RankingConfig()
    ranking_pipeline = RankingPipeline(
        embedder=embedder, judge=judge, config=ranking_config
    )

    if not results.academic_contents:
        print("  ⚠️  No academic content to rank — skipping ranking.")
        results.ranking = RankingResult(papers=[], total_input=0, filtered_by_metadata=0, semantically_ranked=0)
    else:
        print(f"  ⏳ Ranking {len(results.academic_contents)} papers...")
        try:
            ranking_result = await ranking_pipeline.rank(
                results.academic_contents, QUERY
            )
            results.ranking = ranking_result
            results.ranked_papers = ranking_result.papers
            print(f"  ✅ Ranked: {ranking_result.total_input} input → "
                  f"{ranking_result.filtered_by_metadata} filtered → "
                  f"{ranking_result.semantically_ranked} semantic → "
                  f"{len(ranking_result.papers)} final")
        except Exception as exc:
            results.ranking_error = str(exc)
            print(f"  ❌ Ranking failed: {exc}")

    # ------------------------------------------------------------------
    # STEP 4: Profiling
    # ------------------------------------------------------------------
    print_header("STEP 4: PROFILING — Section & claim extraction per paper")
    print("  ⏳ Creating section extractor + profiling pipeline...")

    section_extractor = OpenAISectionExtractor(client=client)
    profile_config = ProfileConfig()
    profiling_pipeline = ProfilingPipeline(
        extractor=section_extractor, config=profile_config
    )

    if not results.ranked_papers:
        print("  ⚠️  No ranked papers to profile — skipping profiling.")
        results.profiling = ProfilingResult(profiles=[], total_input=0)
    else:
        print(f"  ⏳ Profiling {len(results.ranked_papers)} papers...")
        try:
            profiling_result = await profiling_pipeline.profile(
                results.ranked_papers
            )
            results.profiling = profiling_result
            results.profiles = profiling_result.profiles
            total_claims = sum(len(p.claims) for p in profiling_result.profiles)
            print(f"  ✅ Profiled: {len(profiling_result.profiles)} profiles, "
                  f"{total_claims} claims "
                  f"({len(profiling_result.errors)} errors)")
        except Exception as exc:
            results.profiling_error = str(exc)
            print(f"  ❌ Profiling failed: {exc}")

    # ------------------------------------------------------------------
    # STEP 5: Codebase Analysis
    # ------------------------------------------------------------------
    print_header("STEP 5: CODEBASE — Self-analyzing src/ directory")
    print("  ⏳ Scanning AST + generating LLM descriptions...")

    describer = OpenAIDescriber(client=client)
    codebase_config = CodebaseConfig()
    codebase_pipeline = CodebaseAnalysisPipeline(
        describer=describer, config=codebase_config
    )

    try:
        codebase_context = await codebase_pipeline.analyze(CODEBASE_PATH)
        results.codebase = codebase_context
        print(f"  ✅ Codebase: {codebase_context.total_files_scanned} files, "
              f"{len(codebase_context.components)} components, "
              f"{len(codebase_context.modules)} modules")
    except Exception as exc:
        results.codebase_error = str(exc)
        print(f"  ❌ Codebase analysis failed: {exc}")

    # ------------------------------------------------------------------
    # STEP 6: Correlation
    # ------------------------------------------------------------------
    print_header("STEP 6: CORRELATION — Embed → Cosine → Classify → Assemble")
    print("  ⏳ Creating correlation classifier + pipeline (reusing embedder)...")

    corr_classifier = OpenAICorrelationClassifier(client=client)
    corr_config = CorrelationMapConfig()
    correlation_pipeline = CorrelationPipeline(
        embedder=embedder, classifier=corr_classifier, config=corr_config
    )

    if not results.profiles:
        print("  ⚠️  No profiles to correlate — skipping correlation.")
        results.correlation = CorrelationMap(
            papers_analyzed=0,
            total_claims=0,
            total_components=len(results.codebase.components) if results.codebase else 0,
            total_modules=len(results.codebase.modules) if results.codebase else 0,
        )
    elif not results.codebase:
        print("  ⚠️  No codebase context to correlate against — skipping correlation.")
        results.correlation = CorrelationMap(
            papers_analyzed=len(results.profiles),
            total_claims=sum(len(p.claims) for p in results.profiles),
        )
    else:
        print(f"  ⏳ Correlating {len(results.profiles)} profiles × "
              f"{len(results.codebase.components)} components + "
              f"{len(results.codebase.modules)} modules...")
        try:
            correlation_map = await correlation_pipeline.correlate(
                profiles=results.profiles,
                codebase=results.codebase,
            )
            results.correlation = correlation_map
            print(f"  ✅ Correlation: {correlation_map.matches_found} matches "
                  f"({len(correlation_map.correlations)} component + "
                  f"{len(correlation_map.module_correlations)} module)")
        except Exception as exc:
            results.correlation_error = str(exc)
            print(f"  ❌ Correlation failed: {exc}")

    # ------------------------------------------------------------------
    # STEP 7: Hypothesis Generation
    # ------------------------------------------------------------------
    print_header("STEP 7: HYPOTHESIS — Generate → Reflect → Score → Ground → Rank")
    print("  ⏳ Creating hypothesis generator + pipeline (reusing embedder)...")

    hypo_generator = OpenAIHypothesisGenerator(client=client)
    hypo_config = HypothesisConfig()
    hypothesis_pipeline = HypothesisPipeline(
        generator=hypo_generator, embedder=embedder, config=hypo_config
    )

    if not results.correlation:
        print("  ⚠️  No correlation map — skipping hypothesis generation.")
        results.hypothesis = HypothesisResult()
    else:
        print(f"  ⏳ Generating hypotheses from {results.correlation.matches_found} "
              f"correlations...")
        try:
            hypothesis_result = await hypothesis_pipeline.run(
                correlation_map=results.correlation,
                query=QUERY,
                general_content=results.general_contents,
            )
            results.hypothesis = hypothesis_result
            print(f"  ✅ Hypothesis: {len(hypothesis_result.hypotheses)} hypotheses, "
                  f"{hypothesis_result.reflection_rounds_completed} reflection rounds, "
                  f"converged={hypothesis_result.converged}")
        except Exception as exc:
            results.hypothesis_error = str(exc)
            print(f"  ❌ Hypothesis generation failed: {exc}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - t0
    print_full_report(results, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
