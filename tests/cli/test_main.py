"""Tests for the research-to-dev analyze CLI command and orchestrator.

Covers both unit tests (PipelineOrchestrator step logic) and CLI tests
(flag parsing, exit codes, output file handling).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ======================================================================
# Helpers — build realistic mock results for each pipeline step
# ======================================================================


def _make_retrieval_result(n_results: int = 3) -> MagicMock:
    """Build a mock RetrievalResult with the given number of SearchResults."""
    from research_to_dev.retriever.protocol import RetrievalError, RetrievalResult

    results = []
    for i in range(n_results):
        sr = MagicMock()
        sr.source = "arxiv"
        sr.title = f"Paper {i}"
        sr.url = f"http://example.org/{i}"
        sr.abstract = f"Abstract for paper {i}"
        sr.authors = ["Author A"]
        sr.published_date = "2024-01-01"
        sr.metadata = {}
        results.append(sr)
    return RetrievalResult(results=results, errors=[])


def _make_extraction_result(n_academic: int = 3) -> MagicMock:
    """Build a mock ExtractionResult with academic contents."""
    from research_to_dev.extraction.types import (
        AcademicContent,
        ExtractionError,
        ExtractionResult,
    )

    contents = []
    for i in range(n_academic):
        contents.append(
            AcademicContent(
                source="arxiv",
                title=f"Paper {i}",
                url=f"http://example.org/{i}",
                abstract=f"Abstract {i}",
                authors=["Author A"],
            )
        )
    return ExtractionResult(contents=contents, errors=[])


def _make_ranking_result(n_papers: int = 3) -> MagicMock:
    """Build a mock RankingResult with ranked papers."""
    from research_to_dev.extraction.types import AcademicContent
    from research_to_dev.ranking.types import RankedPaper, RankingResult

    papers = []
    for i in range(n_papers):
        ac = AcademicContent(
            source="arxiv",
            title=f"Paper {i}",
            abstract=f"Abstract {i}",
        )
        papers.append(
            RankedPaper(
                paper=ac,
                relevance_score=0.8,
                semantic_score=0.7,
                llm_reasoning="relevant",
                passed_metadata_filter=True,
            )
        )
    return RankingResult(
        papers=papers,
        total_input=n_papers,
        filtered_by_metadata=0,
        semantically_ranked=n_papers,
    )


def _make_profiling_result(n_profiles: int = 3) -> MagicMock:
    """Build a mock ProfilingResult with paper profiles."""
    from research_to_dev.paper_profile.types import (
        Claim,
        PaperProfile,
        ProfilingError,
        ProfilingResult,
    )
    from research_to_dev.ranking.types import RankedPaper
    from research_to_dev.extraction.types import AcademicContent

    profiles = []
    for i in range(n_profiles):
        ac = AcademicContent(source="arxiv", title=f"Paper {i}", abstract=f"Abs {i}")
        rp = RankedPaper(
            paper=ac,
            relevance_score=0.8,
            semantic_score=0.7,
            llm_reasoning="ok",
            passed_metadata_filter=True,
        )
        profiles.append(
            PaperProfile(
                ranked_paper=rp,
                sections=[],
                claims=[
                    Claim(
                        text=f"Claim {i}",
                        paper_id=f"paper-{i}",
                        section_name="abstract",
                    )
                ],
            )
        )
    return ProfilingResult(
        profiles=profiles,
        total_input=n_profiles,
        errors=[],
    )


def _make_codebase_result() -> MagicMock:
    """Build a mock CodebaseContext."""
    from research_to_dev.codebase.types import CodebaseContext, ModuleSummary

    return CodebaseContext(
        project_name="test-project",
        project_path="/tmp/test",
        language="python",
        components=[],
        modules=[
            ModuleSummary(
                module_name="main",
                file_path="/tmp/test/main.py",
                summary="Entry point",
                key_responsibilities=["CLI"],
            )
        ],
        total_files_scanned=5,
        total_components_found=10,
    )


def _make_correlation_result() -> MagicMock:
    """Build a mock CorrelationMap."""
    from research_to_dev.correlation.types import CorrelationMap

    return CorrelationMap(
        correlations=[],
        module_correlations=[],
        papers_analyzed=3,
        total_claims=3,
        total_components=0,
        total_modules=1,
        matches_found=0,
    )


def _make_hypothesis_result(n_hypos: int = 3) -> MagicMock:
    """Build a mock HypothesisResult with hypotheses."""
    from research_to_dev.hypothesis.types import (
        Hypothesis,
        HypothesisError,
        HypothesisResult,
        HypothesisScores,
    )

    hypos = []
    for i in range(n_hypos):
        hypos.append(
            Hypothesis(
                id=f"hash-{i}",
                title=f"Improve X using technique Y ({i})",
                description=f"Description for hypothesis {i}.",
                approach="Implement Z",
                supporting_papers=["Paper A", "Paper B"],
                target_metric="accuracy",
                expected_improvement="higher throughput",
                code_changes="src/main.py",
                scores=HypothesisScores(relevance=8, feasibility=7, evidence=9),
                composite=8.0 - i * 0.5,
                correlations=[],
                success_criteria="p < 0.05",
            )
        )
    return HypothesisResult(
        hypotheses=hypos,
        total_input_papers=3,
        total_correlations=3,
        reflection_rounds_completed=2,
        converged=True,
        errors=[],
    )


# ======================================================================
# Phase 4.1-4.3: Orchestrator unit tests
# ======================================================================


class TestPipelineOrchestratorStepOrdering:
    """Verify the orchestrator calls all 7 steps in the correct order."""

    @pytest.mark.asyncio
    async def test_steps_executed_in_order(self) -> None:
        """All 7 steps are called sequentially: retrieval → ... → hypothesis."""
        from unittest.mock import AsyncMock

        from research_to_dev.cli.orchestrator import PipelineOrchestrator

        client = MagicMock()

        with (
            patch(
                "research_to_dev.cli.orchestrator.RetrieverOrchestrator"
            ) as mock_ret_cls,
            patch(
                "research_to_dev.cli.orchestrator.ExtractionPipeline"
            ) as mock_ext_cls,
            patch(
                "research_to_dev.cli.orchestrator.RankingPipeline"
            ) as mock_rank_cls,
            patch(
                "research_to_dev.cli.orchestrator.ProfilingPipeline"
            ) as mock_prof_cls,
            patch(
                "research_to_dev.cli.orchestrator.CodebaseAnalysisPipeline"
            ) as mock_cb_cls,
            patch(
                "research_to_dev.cli.orchestrator.CorrelationPipeline"
            ) as mock_corr_cls,
            patch(
                "research_to_dev.cli.orchestrator.HypothesisPipeline"
            ) as mock_hypo_cls,
            patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True),
        ):
            # Setup mocks
            mock_ret = MagicMock()
            mock_ret.search = AsyncMock(return_value=_make_retrieval_result(3))
            mock_ret_cls.return_value = mock_ret

            mock_ext = MagicMock()
            mock_ext.extract = AsyncMock(return_value=_make_extraction_result(3))
            mock_ext_cls.return_value = mock_ext

            mock_rank = MagicMock()
            mock_rank.rank = AsyncMock(return_value=_make_ranking_result(3))
            mock_rank_cls.return_value = mock_rank

            mock_prof = MagicMock()
            mock_prof.profile = AsyncMock(return_value=_make_profiling_result(3))
            mock_prof_cls.return_value = mock_prof

            mock_cb = MagicMock()
            mock_cb.analyze = AsyncMock(return_value=_make_codebase_result())
            mock_cb_cls.return_value = mock_cb

            mock_corr = MagicMock()
            mock_corr.correlate = AsyncMock(return_value=_make_correlation_result())
            mock_corr_cls.return_value = mock_corr

            mock_hypo = MagicMock()
            mock_hypo.run = AsyncMock(return_value=_make_hypothesis_result(3))
            mock_hypo_cls.return_value = mock_hypo

            orchestrator = PipelineOrchestrator(client)
            trace = await orchestrator.run("test query", "/tmp/test")

            # All steps should be called
            mock_ret.search.assert_called_once()
            mock_ext.extract.assert_called_once()
            mock_rank.rank.assert_called_once()
            mock_prof.profile.assert_called_once()
            mock_cb.analyze.assert_called_once()
            mock_corr.correlate.assert_called_once()
            mock_hypo.run.assert_called_once()

            # Trace should contain all 7 step entries
            assert len(trace.steps) == 7
            assert "retrieval" in trace.steps
            assert "extraction" in trace.steps
            assert "ranking" in trace.steps
            assert "profiling" in trace.steps
            assert "codebase" in trace.steps
            assert "correlation" in trace.steps
            assert "hypothesis" in trace.steps

    @pytest.mark.asyncio
    async def test_happy_path_all_steps_success(self) -> None:
        """On happy path, all steps have status 'success'."""
        from unittest.mock import AsyncMock

        from research_to_dev.cli.orchestrator import PipelineOrchestrator

        client = MagicMock()

        with (
            patch(
                "research_to_dev.cli.orchestrator.RetrieverOrchestrator"
            ) as mock_ret_cls,
            patch(
                "research_to_dev.cli.orchestrator.ExtractionPipeline"
            ) as mock_ext_cls,
            patch(
                "research_to_dev.cli.orchestrator.RankingPipeline"
            ) as mock_rank_cls,
            patch(
                "research_to_dev.cli.orchestrator.ProfilingPipeline"
            ) as mock_prof_cls,
            patch(
                "research_to_dev.cli.orchestrator.CodebaseAnalysisPipeline"
            ) as mock_cb_cls,
            patch(
                "research_to_dev.cli.orchestrator.CorrelationPipeline"
            ) as mock_corr_cls,
            patch(
                "research_to_dev.cli.orchestrator.HypothesisPipeline"
            ) as mock_hypo_cls,
            patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True),
        ):
            mock_ret_cls.return_value.search = AsyncMock(
                return_value=_make_retrieval_result(3)
            )
            mock_ext_cls.return_value.extract = AsyncMock(
                return_value=_make_extraction_result(3)
            )
            mock_rank_cls.return_value.rank = AsyncMock(
                return_value=_make_ranking_result(3)
            )
            mock_prof_cls.return_value.profile = AsyncMock(
                return_value=_make_profiling_result(3)
            )
            mock_cb_cls.return_value.analyze = AsyncMock(
                return_value=_make_codebase_result()
            )
            mock_corr_cls.return_value.correlate = AsyncMock(
                return_value=_make_correlation_result()
            )
            mock_hypo_cls.return_value.run = AsyncMock(
                return_value=_make_hypothesis_result(3)
            )

            orchestrator = PipelineOrchestrator(client)
            trace = await orchestrator.run("test", "/tmp/test")

            for step_name, step_trace in trace.steps.items():
                assert step_trace.status == "success", (
                    f"Step '{step_name}' has status '{step_trace.status}', "
                    f"expected 'success'"
                )


class TestPipelineOrchestratorFailFast:
    """Verify fail-fast logic: empty results stop the pipeline."""

    @pytest.mark.asyncio
    async def test_empty_retrieval_stops_pipeline(self) -> None:
        """Empty retrieval → status 'empty', no downstream steps called."""
        from unittest.mock import AsyncMock

        from research_to_dev.cli.orchestrator import PipelineOrchestrator

        client = MagicMock()

        with (
            patch(
                "research_to_dev.cli.orchestrator.RetrieverOrchestrator"
            ) as mock_ret_cls,
            patch(
                "research_to_dev.cli.orchestrator.ExtractionPipeline"
            ) as mock_ext_cls,
            patch(
                "research_to_dev.cli.orchestrator.RankingPipeline"
            ) as mock_rank_cls,
            patch(
                "research_to_dev.cli.orchestrator.ProfilingPipeline"
            ) as mock_prof_cls,
            patch(
                "research_to_dev.cli.orchestrator.CodebaseAnalysisPipeline"
            ) as mock_cb_cls,
            patch(
                "research_to_dev.cli.orchestrator.CorrelationPipeline"
            ) as mock_corr_cls,
            patch(
                "research_to_dev.cli.orchestrator.HypothesisPipeline"
            ) as mock_hypo_cls,
            patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True),
        ):
            mock_ret_cls.return_value.search = AsyncMock(
                return_value=_make_retrieval_result(0)
            )  # ZERO results

            orchestrator = PipelineOrchestrator(client)
            trace = await orchestrator.run("test", "/tmp/test")

            # Retrieval step should be "empty"
            assert trace.steps["retrieval"].status == "empty"

            # No downstream steps should be called
            mock_ext_cls.return_value.extract.assert_not_called()
            mock_rank_cls.return_value.rank.assert_not_called()
            mock_prof_cls.return_value.profile.assert_not_called()
            mock_cb_cls.return_value.analyze.assert_not_called()
            mock_corr_cls.return_value.correlate.assert_not_called()
            mock_hypo_cls.return_value.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_extraction_stops_pipeline(self) -> None:
        """Empty extraction (0 contents) → status 'empty', stops."""
        from unittest.mock import AsyncMock

        from research_to_dev.cli.orchestrator import PipelineOrchestrator
        from research_to_dev.extraction.types import ExtractionResult

        client = MagicMock()

        with (
            patch(
                "research_to_dev.cli.orchestrator.RetrieverOrchestrator"
            ) as mock_ret_cls,
            patch(
                "research_to_dev.cli.orchestrator.ExtractionPipeline"
            ) as mock_ext_cls,
            patch(
                "research_to_dev.cli.orchestrator.RankingPipeline"
            ) as mock_rank_cls,
            patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True),
        ):
            mock_ret_cls.return_value.search = AsyncMock(
                return_value=_make_retrieval_result(3)
            )
            mock_ext_cls.return_value.extract = AsyncMock(
                return_value=ExtractionResult(contents=[], errors=[])
            )

            orchestrator = PipelineOrchestrator(client)
            trace = await orchestrator.run("test", "/tmp/test")

            assert trace.steps["retrieval"].status == "success"
            assert trace.steps["extraction"].status == "empty"
            mock_rank_cls.return_value.rank.assert_not_called()


class TestPipelineOrchestratorPartialFailure:
    """Verify graceful degradation: partial failures continue downstream."""

    @pytest.mark.asyncio
    async def test_ranking_error_continues_pipeline(self) -> None:
        """Ranking step error → status 'error', profiling continues."""
        from unittest.mock import AsyncMock

        from research_to_dev.cli.orchestrator import PipelineOrchestrator

        client = MagicMock()

        with (
            patch(
                "research_to_dev.cli.orchestrator.RetrieverOrchestrator"
            ) as mock_ret_cls,
            patch(
                "research_to_dev.cli.orchestrator.ExtractionPipeline"
            ) as mock_ext_cls,
            patch(
                "research_to_dev.cli.orchestrator.RankingPipeline"
            ) as mock_rank_cls,
            patch(
                "research_to_dev.cli.orchestrator.ProfilingPipeline"
            ) as mock_prof_cls,
            patch(
                "research_to_dev.cli.orchestrator.CodebaseAnalysisPipeline"
            ) as mock_cb_cls,
            patch(
                "research_to_dev.cli.orchestrator.CorrelationPipeline"
            ) as mock_corr_cls,
            patch(
                "research_to_dev.cli.orchestrator.HypothesisPipeline"
            ) as mock_hypo_cls,
            patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True),
        ):
            mock_ret_cls.return_value.search = AsyncMock(
                return_value=_make_retrieval_result(3)
            )
            mock_ext_cls.return_value.extract = AsyncMock(
                return_value=_make_extraction_result(3)
            )
            mock_rank_cls.return_value.rank = AsyncMock(
                side_effect=RuntimeError("LLM timeout")
            )
            mock_prof_cls.return_value.profile = AsyncMock(
                return_value=_make_profiling_result(3)
            )
            mock_cb_cls.return_value.analyze = AsyncMock(
                return_value=_make_codebase_result()
            )
            mock_corr_cls.return_value.correlate = AsyncMock(
                return_value=_make_correlation_result()
            )
            mock_hypo_cls.return_value.run = AsyncMock(
                return_value=_make_hypothesis_result(3)
            )

            orchestrator = PipelineOrchestrator(client)
            trace = await orchestrator.run("test", "/tmp/test")

            # Ranking should show error
            assert trace.steps["ranking"].status == "error"
            assert "LLM timeout" in trace.steps["ranking"].error

            # Downstream steps should still run (profiling skipped due to no ranked papers)
            mock_cb_cls.return_value.analyze.assert_called_once()

            # Warnings should include the ranking error
            assert any("ranking" in w for w in trace.warnings)

    @pytest.mark.asyncio
    async def test_codebase_error_continues_pipeline(self) -> None:
        """Codebase step error → status 'error', correlation continues."""
        from unittest.mock import AsyncMock

        from research_to_dev.cli.orchestrator import PipelineOrchestrator

        client = MagicMock()

        with (
            patch(
                "research_to_dev.cli.orchestrator.RetrieverOrchestrator"
            ) as mock_ret_cls,
            patch(
                "research_to_dev.cli.orchestrator.ExtractionPipeline"
            ) as mock_ext_cls,
            patch(
                "research_to_dev.cli.orchestrator.RankingPipeline"
            ) as mock_rank_cls,
            patch(
                "research_to_dev.cli.orchestrator.ProfilingPipeline"
            ) as mock_prof_cls,
            patch(
                "research_to_dev.cli.orchestrator.CodebaseAnalysisPipeline"
            ) as mock_cb_cls,
            patch(
                "research_to_dev.cli.orchestrator.CorrelationPipeline"
            ) as mock_corr_cls,
            patch(
                "research_to_dev.cli.orchestrator.HypothesisPipeline"
            ) as mock_hypo_cls,
            patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True),
        ):
            mock_ret_cls.return_value.search = AsyncMock(
                return_value=_make_retrieval_result(3)
            )
            mock_ext_cls.return_value.extract = AsyncMock(
                return_value=_make_extraction_result(3)
            )
            mock_rank_cls.return_value.rank = AsyncMock(
                return_value=_make_ranking_result(3)
            )
            mock_prof_cls.return_value.profile = AsyncMock(
                return_value=_make_profiling_result(3)
            )
            mock_cb_cls.return_value.analyze = AsyncMock(
                side_effect=RuntimeError("AST parse failed")
            )
            mock_corr_cls.return_value.correlate = AsyncMock(
                return_value=_make_correlation_result()
            )
            mock_hypo_cls.return_value.run = AsyncMock(
                return_value=_make_hypothesis_result(3)
            )

            orchestrator = PipelineOrchestrator(client)
            trace = await orchestrator.run("test", "/tmp/test")

            assert trace.steps["codebase"].status == "error"
            assert any("codebase" in w for w in trace.warnings)
            # Correlation is skipped when codebase is None (nothing to correlate)
            assert trace.steps.get("correlation") is not None
            assert trace.steps["correlation"].status == "skipped"


# ======================================================================
# Phase 4.4-4.8: CLI tests
# ======================================================================


class TestCliFlagParsing:
    """Verify CLI flag parsing and validation."""

    def test_analyze_help_shows_flags(self) -> None:
        """'analyze --help' shows expected flags."""
        from research_to_dev.cli.main import app

        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--query" in result.stdout
        assert "--codebase" in result.stdout
        assert "--output" in result.stdout

    def test_root_help_shows_analyze(self) -> None:
        """'--help' shows the 'analyze' command."""
        from research_to_dev.cli.main import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.stdout
        # codebase subcommand should NOT be present
        assert "codebase" not in result.stdout

    def test_missing_query_shows_error(self) -> None:
        """Invoking without --query shows an error."""
        from research_to_dev.cli.main import app

        result = runner.invoke(app, ["analyze"])
        assert result.exit_code != 0

    def test_default_values(self) -> None:
        """--codebase and --output have correct defaults."""
        from research_to_dev.cli.main import app

        # Just check that defaults are documented
        result = runner.invoke(app, ["analyze", "--help"])
        assert "[default: .]" in result.stdout or "default: ." in result.stdout
        assert (
            "[default: results.json]" in result.stdout
            or "default: results.json" in result.stdout
        )


class TestCliApiKeyValidation:
    """Verify OPENAI_API_KEY validation before pipeline start."""

    def test_missing_api_key_exits_with_code_1(self) -> None:
        """Missing OPENAI_API_KEY → exit code 1 with clear message."""
        from research_to_dev.cli.main import app

        with patch.dict(os.environ, {}, clear=True):
            with patch("research_to_dev.cli.main.load_dotenv"):
                result = runner.invoke(
                    app, ["analyze", "--query", "test", "--codebase", "/tmp"]
                )
            assert result.exit_code == 1
            assert "OPENAI_API_KEY" in result.output

    def test_api_key_present_proceeds(self) -> None:
        """With OPENAI_API_KEY set, the command proceeds past validation."""
        from research_to_dev.cli.main import app

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("research_to_dev.cli.main.load_dotenv"),
            patch("research_to_dev.cli.main.AsyncOpenAI") as mock_ai,
            patch(
                "research_to_dev.cli.main.PipelineOrchestrator"
            ) as mock_orch,
        ):
            mock_orch.return_value.run = AsyncMock(
                return_value=MagicMock(
                    steps={"retrieval": MagicMock(status="success")},
                    to_json=MagicMock(return_value="{}"),
                )
            )

            # Use a valid temp directory for --codebase
            with tempfile.TemporaryDirectory() as tmpdir:
                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        "--query",
                        "test",
                        "--codebase",
                        tmpdir,
                        "--output",
                        f"{tmpdir}/out.json",
                    ],
                )
                # Should NOT fail at API key stage
                assert mock_ai.called or mock_orch.called


class TestCliPathValidation:
    """Verify codebase path validation."""

    def test_invalid_codebase_path_exits_with_code_1(self) -> None:
        """Non-existent --codebase path → exit code 1."""
        from research_to_dev.cli.main import app

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("research_to_dev.cli.main.load_dotenv"),
        ):
            result = runner.invoke(
                app,
                [
                    "analyze",
                    "--query",
                    "test",
                    "--codebase",
                    "/nonexistent/path/xyz",
                ],
            )
            assert result.exit_code == 1
            assert "not a valid directory" in result.output


class TestCliOutputFile:
    """Verify JSON output file creation and content."""

    def test_output_file_created_with_valid_json(self) -> None:
        """Pipeline produces a valid JSON file matching spec keys."""
        from research_to_dev.cli.main import app
        from research_to_dev.cli.orchestrator import PipelineTrace, StepTrace

        trace = PipelineTrace(
            query="test query",
            codebase_path="/tmp/test",
            timestamp="2024-01-01T00:00:00+00:00",
            steps={
                "retrieval": StepTrace(status="success"),
                "extraction": StepTrace(status="success"),
                "ranking": StepTrace(status="success"),
                "profiling": StepTrace(status="success"),
                "codebase": StepTrace(status="success"),
                "correlation": StepTrace(status="success"),
                "hypothesis": StepTrace(status="success"),
            },
            hypotheses=[
                {
                    "title": "Test Hypothesis",
                    "composite": 8.5,
                    "description": "A test",
                    "supporting_papers": ["Paper A"],
                }
            ],
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("research_to_dev.cli.main.load_dotenv"),
            patch("research_to_dev.cli.main.AsyncOpenAI"),
            patch(
                "research_to_dev.cli.main.PipelineOrchestrator"
            ) as mock_orch,
        ):
            mock_orch.return_value.run = AsyncMock(return_value=trace)

            with tempfile.TemporaryDirectory() as tmpdir:
                outpath = f"{tmpdir}/results.json"
                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        "--query",
                        "test",
                        "--codebase",
                        tmpdir,
                        "--output",
                        outpath,
                    ],
                )
                assert result.exit_code == 0

                # Verify file exists and is valid JSON
                assert Path(outpath).exists()
                data = json.loads(Path(outpath).read_text())
                assert data["query"] == "test query"
                assert data["codebase_path"] == "/tmp/test"
                assert "steps" in data
                assert "hypotheses" in data
                assert len(data["hypotheses"]) == 1

    def test_output_directory_created_if_missing(self) -> None:
        """--output with non-existent parent dir creates it."""
        from research_to_dev.cli.main import app
        from research_to_dev.cli.orchestrator import PipelineTrace, StepTrace

        trace = PipelineTrace(
            query="test",
            codebase_path="/tmp/test",
            timestamp="2024-01-01T00:00:00+00:00",
            steps={"retrieval": StepTrace(status="success")},
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("research_to_dev.cli.main.load_dotenv"),
            patch("research_to_dev.cli.main.AsyncOpenAI"),
            patch(
                "research_to_dev.cli.main.PipelineOrchestrator"
            ) as mock_orch,
        ):
            mock_orch.return_value.run = AsyncMock(return_value=trace)

            with tempfile.TemporaryDirectory() as tmpdir:
                outpath = f"{tmpdir}/subdir/deep/results.json"
                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        "--query",
                        "test",
                        "--codebase",
                        tmpdir,
                        "--output",
                        outpath,
                    ],
                )
                assert result.exit_code == 0
                assert Path(outpath).exists()

    def test_output_file_overwrites_existing(self) -> None:
        """Existing output file is overwritten with fresh pipeline trace JSON."""
        from research_to_dev.cli.main import app
        from research_to_dev.cli.orchestrator import PipelineTrace, StepTrace

        trace = PipelineTrace(
            query="fresh query",
            codebase_path="/tmp/test",
            timestamp="2024-06-04T00:00:00+00:00",
            steps={
                "retrieval": StepTrace(status="success"),
                "extraction": StepTrace(status="success"),
                "ranking": StepTrace(status="success"),
                "profiling": StepTrace(status="success"),
                "codebase": StepTrace(status="success"),
                "correlation": StepTrace(status="success"),
                "hypothesis": StepTrace(status="success"),
            },
            hypotheses=[
                {
                    "title": "Overwritten Hypothesis",
                    "composite": 9.0,
                    "description": "Fresh result",
                    "supporting_papers": ["Paper X"],
                }
            ],
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("research_to_dev.cli.main.load_dotenv"),
            patch("research_to_dev.cli.main.AsyncOpenAI"),
            patch(
                "research_to_dev.cli.main.PipelineOrchestrator"
            ) as mock_orch,
        ):
            mock_orch.return_value.run = AsyncMock(return_value=trace)

            with tempfile.TemporaryDirectory() as tmpdir:
                outpath = f"{tmpdir}/output.json"
                # Pre-create the file with old content
                Path(outpath).write_text('{"old": "data"}')

                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        "--query",
                        "fresh query",
                        "--codebase",
                        tmpdir,
                        "--output",
                        outpath,
                    ],
                )
                assert result.exit_code == 0

                # Verify the file was overwritten
                data = json.loads(Path(outpath).read_text())
                assert data["query"] == "fresh query"
                assert "steps" in data
                assert "old" not in data  # old content gone
                assert data["hypotheses"][0]["title"] == "Overwritten Hypothesis"


class TestCliTerminalSummary:
    """Verify terminal summary output format."""

    def test_summary_contains_hypothesis_table(self) -> None:
        """Terminal output contains top hypotheses table."""
        from research_to_dev.cli.main import app
        from research_to_dev.cli.orchestrator import PipelineTrace, StepTrace

        trace = PipelineTrace(
            query="test query",
            codebase_path="/tmp/test",
            timestamp="2024-01-01T00:00:00+00:00",
            steps={
                "retrieval": StepTrace(status="success"),
                "extraction": StepTrace(status="success"),
                "ranking": StepTrace(status="success"),
                "profiling": StepTrace(status="success"),
                "codebase": StepTrace(status="success"),
                "correlation": StepTrace(status="success"),
                "hypothesis": StepTrace(status="success"),
            },
            hypotheses=[
                {
                    "title": "Improve performance with caching",
                    "composite": 9.2,
                    "description": "A test",
                    "supporting_papers": ["Paper A"],
                },
            ],
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("research_to_dev.cli.main.load_dotenv"),
            patch("research_to_dev.cli.main.AsyncOpenAI"),
            patch(
                "research_to_dev.cli.main.PipelineOrchestrator"
            ) as mock_orch,
        ):
            mock_orch.return_value.run = AsyncMock(return_value=trace)

            with tempfile.TemporaryDirectory() as tmpdir:
                outpath = f"{tmpdir}/results.json"
                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        "--query",
                        "test",
                        "--codebase",
                        tmpdir,
                        "--output",
                        outpath,
                    ],
                )
                assert result.exit_code == 0
                assert "PIPELINE RESULTS" in result.stdout
                assert "Top Hypotheses" in result.stdout
                assert "Improve performance with caching" in result.stdout
                assert "9.2" in result.stdout
                assert "Output:" in result.stdout

    def test_summary_shows_no_hypotheses_message(self) -> None:
        """When no hypotheses, terminal shows appropriate message."""
        from research_to_dev.cli.main import app
        from research_to_dev.cli.orchestrator import PipelineTrace, StepTrace

        trace = PipelineTrace(
            query="test query",
            codebase_path="/tmp/test",
            timestamp="2024-01-01T00:00:00+00:00",
            steps={
                "retrieval": StepTrace(status="success"),
                "hypothesis": StepTrace(status="success"),
            },
            hypotheses=[],
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("research_to_dev.cli.main.load_dotenv"),
            patch("research_to_dev.cli.main.AsyncOpenAI"),
            patch(
                "research_to_dev.cli.main.PipelineOrchestrator"
            ) as mock_orch,
        ):
            mock_orch.return_value.run = AsyncMock(return_value=trace)

            with tempfile.TemporaryDirectory() as tmpdir:
                outpath = f"{tmpdir}/results.json"
                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        "--query",
                        "test",
                        "--codebase",
                        tmpdir,
                        "--output",
                        outpath,
                    ],
                )
                assert result.exit_code == 0
                assert "No hypotheses generated" in result.stdout

    def test_summary_shows_warnings(self) -> None:
        """Terminal summary includes warnings section when warnings exist."""
        from research_to_dev.cli.main import app
        from research_to_dev.cli.orchestrator import PipelineTrace, StepTrace

        trace = PipelineTrace(
            query="test",
            codebase_path="/tmp/test",
            timestamp="2024-01-01T00:00:00+00:00",
            steps={
                "retrieval": StepTrace(
                    status="success",
                    warnings=["Source X failed", "Source Y timed out"],
                ),
            },
            warnings=["Source X failed", "Source Y timed out"],
        )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("research_to_dev.cli.main.load_dotenv"),
            patch("research_to_dev.cli.main.AsyncOpenAI"),
            patch(
                "research_to_dev.cli.main.PipelineOrchestrator"
            ) as mock_orch,
        ):
            mock_orch.return_value.run = AsyncMock(return_value=trace)

            with tempfile.TemporaryDirectory() as tmpdir:
                outpath = f"{tmpdir}/results.json"
                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        "--query",
                        "test",
                        "--codebase",
                        tmpdir,
                        "--output",
                        outpath,
                    ],
                )
                assert result.exit_code == 0
                assert "Warnings:" in result.stdout
                assert "2" in result.stdout  # count


# ======================================================================
# Phase 4.9: Callback tests
# ======================================================================


class TestPipelineOrchestratorCallback:
    """Verify _on_step callback behaviour."""

    @pytest.mark.asyncio
    async def test_on_step_callback_invoked_with_step_numbers(self) -> None:
        """Callback receives messages with correct step numbers 1-7."""
        from unittest.mock import AsyncMock

        from research_to_dev.cli.orchestrator import PipelineOrchestrator

        client = MagicMock()
        calls: list[tuple[str, int]] = []

        def on_step(msg: str, step: int) -> None:
            calls.append((msg, step))

        orchestrator = PipelineOrchestrator(client, on_step=on_step)

        with (
            patch(
                "research_to_dev.cli.orchestrator.RetrieverOrchestrator"
            ) as mock_ret_cls,
            patch(
                "research_to_dev.cli.orchestrator.ExtractionPipeline"
            ) as mock_ext_cls,
            patch(
                "research_to_dev.cli.orchestrator.RankingPipeline"
            ) as mock_rank_cls,
            patch(
                "research_to_dev.cli.orchestrator.ProfilingPipeline"
            ) as mock_prof_cls,
            patch(
                "research_to_dev.cli.orchestrator.CodebaseAnalysisPipeline"
            ) as mock_cb_cls,
            patch(
                "research_to_dev.cli.orchestrator.CorrelationPipeline"
            ) as mock_corr_cls,
            patch(
                "research_to_dev.cli.orchestrator.HypothesisPipeline"
            ) as mock_hypo_cls,
            patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True),
        ):
            mock_ret_cls.return_value.search = AsyncMock(
                return_value=_make_retrieval_result(3)
            )
            mock_ext_cls.return_value.extract = AsyncMock(
                return_value=_make_extraction_result(3)
            )
            mock_rank_cls.return_value.rank = AsyncMock(
                return_value=_make_ranking_result(3)
            )
            mock_prof_cls.return_value.profile = AsyncMock(
                return_value=_make_profiling_result(3)
            )
            mock_cb_cls.return_value.analyze = AsyncMock(
                return_value=_make_codebase_result()
            )
            mock_corr_cls.return_value.correlate = AsyncMock(
                return_value=_make_correlation_result()
            )
            mock_hypo_cls.return_value.run = AsyncMock(
                return_value=_make_hypothesis_result(3)
            )

            await orchestrator.run("test query", "/tmp/test")

        # Callback was invoked at least once
        assert len(calls) > 0, "Expected _on_step to be called at least once"

        # Step numbers 1 and 7 should be present
        step_nums = [c[1] for c in calls]
        assert 1 in step_nums, "Expected step 1 in callbacks"
        assert 7 in step_nums, "Expected step 7 in callbacks"

        # At least one message contains expected text
        search_messages = [c[0] for c in calls if "papers" in c[0].lower()]
        assert len(search_messages) > 0, (
            "Expected at least one callback message mentioning 'papers'"
        )

    @pytest.mark.asyncio
    async def test_on_step_none_no_errors(self) -> None:
        """When on_step=None, orchestrator runs without errors."""
        from unittest.mock import AsyncMock

        from research_to_dev.cli.orchestrator import PipelineOrchestrator

        client = MagicMock()

        # No callback
        orchestrator = PipelineOrchestrator(client, on_step=None)

        with (
            patch(
                "research_to_dev.cli.orchestrator.RetrieverOrchestrator"
            ) as mock_ret_cls,
            patch(
                "research_to_dev.cli.orchestrator.ExtractionPipeline"
            ) as mock_ext_cls,
            patch(
                "research_to_dev.cli.orchestrator.RankingPipeline"
            ) as mock_rank_cls,
            patch(
                "research_to_dev.cli.orchestrator.ProfilingPipeline"
            ) as mock_prof_cls,
            patch(
                "research_to_dev.cli.orchestrator.CodebaseAnalysisPipeline"
            ) as mock_cb_cls,
            patch(
                "research_to_dev.cli.orchestrator.CorrelationPipeline"
            ) as mock_corr_cls,
            patch(
                "research_to_dev.cli.orchestrator.HypothesisPipeline"
            ) as mock_hypo_cls,
            patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=True),
        ):
            mock_ret_cls.return_value.search = AsyncMock(
                return_value=_make_retrieval_result(3)
            )
            mock_ext_cls.return_value.extract = AsyncMock(
                return_value=_make_extraction_result(3)
            )
            mock_rank_cls.return_value.rank = AsyncMock(
                return_value=_make_ranking_result(3)
            )
            mock_prof_cls.return_value.profile = AsyncMock(
                return_value=_make_profiling_result(3)
            )
            mock_cb_cls.return_value.analyze = AsyncMock(
                return_value=_make_codebase_result()
            )
            mock_corr_cls.return_value.correlate = AsyncMock(
                return_value=_make_correlation_result()
            )
            mock_hypo_cls.return_value.run = AsyncMock(
                return_value=_make_hypothesis_result(3)
            )

            # Should not raise
            trace = await orchestrator.run("test query", "/tmp/test")

        # Pipeline should have completed all 7 steps
        assert len(trace.steps) == 7
        for step_trace in trace.steps.values():
            assert step_trace.status == "success"
