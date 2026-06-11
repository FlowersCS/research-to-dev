"""Tests for insights dataclasses, Protocol, OpenAI adapter, formatters,
and CLI integration with --with-insights.

Uses ``AsyncMock`` to verify API call shapes without real API calls.
No ``OPENAI_API_KEY`` required.
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from research_to_dev.cli.main import app
from research_to_dev.report.compilation import (
    BaselineComparison,
    CompiledReport,
    HypothesisSummary,
    IterationResult,
    TsvResultsReader,
    compile_all,
    format_json,
    format_markdown,
)
from research_to_dev.report.insights import (
    EvidenceCorrelation,
    InsightsGenerator,
    InsightsReport,
    OpenAIInsightsGenerator,
    PatternInsight,
    ProgramContext,
    Recommendation,
)


# ===================================================================
# Helpers
# ===================================================================


def _mock_insights_response() -> AsyncMock:
    """Build an AsyncMock response with valid JSON containing all three
    sections."""
    mock_choice = AsyncMock()
    mock_message = AsyncMock()
    mock_message.content = json.dumps(
        {
            "patterns": [
                {
                    "title": "Consistent latency improvement",
                    "description": "Multiple hypotheses show latency reduction.",
                    "affected_hypotheses": ["h1", "h2"],
                    "confidence": "high",
                }
            ],
            "recommendations": [
                {
                    "action": "Adopt batched inference",
                    "priority": "high",
                    "rationale": "3 of 5 hypotheses show improvement.",
                    "supporting_evidence": ["h1", "h2"],
                }
            ],
            "evidence_correlation": [
                {
                    "paper_id": "Smith et al. 2024",
                    "evidence_summary": "Batching reduces latency.",
                    "related_hypotheses": ["h1"],
                    "code_areas": ["src/inference/engine.py"],
                }
            ],
        }
    )
    mock_choice.message = mock_message
    mock_response = AsyncMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _mock_hypothesis_summaries() -> list[HypothesisSummary]:
    """Return sample HypothesisSummary instances for testing."""
    return [
        HypothesisSummary(
            hypothesis_id="h1",
            status="improved",
            iterations_total=3,
            iterations_kept=2,
            iterations_discarded=0,
            iterations_crashed=1,
            best_metric_value=0.42,
            baseline_comparison=BaselineComparison(
                best_improvement=-0.05,
                overall_trend="improved",
            ),
            iterations=[
                IterationResult(
                    iteration=1,
                    metric_name="val_loss",
                    metric_value=0.45,
                    baseline_value=0.50,
                    delta=-0.05,
                    status="success",
                    timestamp="2026-06-07T14:00:00Z",
                ),
            ],
        ),
        HypothesisSummary(
            hypothesis_id="h2",
            status="improved",
            iterations_total=2,
            iterations_kept=2,
            iterations_discarded=0,
            iterations_crashed=0,
            best_metric_value=0.95,
            baseline_comparison=BaselineComparison(
                best_improvement=0.03,
                overall_trend="improved",
            ),
            iterations=[
                IterationResult(
                    iteration=1,
                    metric_name="accuracy",
                    metric_value=0.95,
                    baseline_value=0.92,
                    delta=0.03,
                    status="success",
                    timestamp="2026-06-07T14:00:00Z",
                ),
            ],
        ),
    ]


def _mock_program_contexts() -> list[ProgramContext]:
    """Return sample ProgramContext instances for testing."""
    return [
        ProgramContext(
            hypothesis_id="h1",
            direction="minimize",
            baseline="val_loss=0.5",
            success_criteria="val_loss < 0.3",
        ),
        ProgramContext(
            hypothesis_id="h2",
            direction="maximize",
            baseline="accuracy=0.92",
            success_criteria="accuracy >= 0.95",
        ),
    ]


def _sample_insights_report() -> InsightsReport:
    """Return an InsightsReport with 1 pattern, 2 recommendations,
    and 1 evidence correlation."""
    return InsightsReport(
        patterns=[
            PatternInsight(
                title="Consistent latency improvement",
                description="Multiple hypotheses targeting latency showed gains.",
                affected_hypotheses=["h1", "h2"],
                confidence="high",
            ),
        ],
        recommendations=[
            Recommendation(
                action="Adopt batching",
                priority="high",
                rationale="Batching helps.",
                supporting_evidence=["h1"],
            ),
            Recommendation(
                action="Increase logging",
                priority="low",
                rationale="Better debugging.",
                supporting_evidence=["h2"],
            ),
        ],
        evidence_correlation=[
            EvidenceCorrelation(
                paper_id="Smith et al. 2024",
                evidence_summary="Batching reduces latency by 40%.",
                related_hypotheses=["h1"],
                code_areas=["src/inference/engine.py"],
            ),
        ],
    )


# ===================================================================
# FR-08: ReportConfig insights_model field
# ===================================================================


class TestReportConfigInsightsModel:
    """FR-08: ReportConfig.insights_model default and custom values."""

    def test_default_insights_model(self) -> None:
        """Default insights_model is 'gpt-4o-mini' (FR-08, 08-A)."""
        from research_to_dev.shared.config import ReportConfig

        config = ReportConfig()
        assert config.insights_model == "gpt-4o-mini"

    def test_custom_insights_model(self) -> None:
        """Custom insights_model is accepted via constructor (FR-08, 08-B)."""
        from research_to_dev.shared.config import ReportConfig

        config = ReportConfig(insights_model="gpt-4o")
        assert config.insights_model == "gpt-4o"


# ===================================================================
# LI-11: Dataclass and Protocol unit tests
# ===================================================================


class TestPatternInsight:
    """Construction and field access for PatternInsight."""

    def test_all_fields_populated(self) -> None:
        """All fields accessible after construction."""
        pi = PatternInsight(
            title="Consistent latency improvement",
            description="Multiple hypotheses show latency reduction.",
            affected_hypotheses=["h1", "h2"],
            confidence="high",
        )
        assert pi.title == "Consistent latency improvement"
        assert pi.description == "Multiple hypotheses show latency reduction."
        assert pi.affected_hypotheses == ["h1", "h2"]
        assert pi.confidence == "high"


class TestRecommendation:
    """Construction and field access for Recommendation."""

    def test_all_fields_populated(self) -> None:
        """All fields accessible after construction."""
        rec = Recommendation(
            action="Adopt batched inference",
            priority="high",
            rationale="3 of 5 hypotheses show improvement.",
            supporting_evidence=["h1", "h2"],
        )
        assert rec.action == "Adopt batched inference"
        assert rec.priority == "high"
        assert rec.rationale == "3 of 5 hypotheses show improvement."
        assert rec.supporting_evidence == ["h1", "h2"]


class TestEvidenceCorrelation:
    """Construction and field access for EvidenceCorrelation."""

    def test_all_fields_populated(self) -> None:
        """All fields accessible after construction."""
        ec = EvidenceCorrelation(
            paper_id="Smith et al. 2024",
            evidence_summary="Batching reduces latency by 40%.",
            related_hypotheses=["h1"],
            code_areas=["src/inference/engine.py"],
        )
        assert ec.paper_id == "Smith et al. 2024"
        assert ec.evidence_summary == "Batching reduces latency by 40%."
        assert ec.related_hypotheses == ["h1"]
        assert ec.code_areas == ["src/inference/engine.py"]


class TestInsightsReport:
    """Construction for InsightsReport container."""

    def test_construction_with_all_sections(self) -> None:
        """Constructed with all three sections."""
        p = PatternInsight(
            title="T",
            description="D",
            affected_hypotheses=["h1"],
            confidence="medium",
        )
        r = Recommendation(
            action="A",
            priority="low",
            rationale="R",
            supporting_evidence=["h1"],
        )
        ec = EvidenceCorrelation(
            paper_id="P",
            evidence_summary="E",
            related_hypotheses=["h1"],
            code_areas=["c"],
        )
        report = InsightsReport(
            patterns=[p],
            recommendations=[r],
            evidence_correlation=[ec],
        )
        assert report.patterns == [p]
        assert report.recommendations == [r]
        assert report.evidence_correlation == [ec]

    def test_construction_with_empty_defaults(self) -> None:
        """InsightsReport() produces all empty lists."""
        report = InsightsReport()
        assert report.patterns == []
        assert report.recommendations == []
        assert report.evidence_correlation == []


class TestProgramContext:
    """Construction and field access for ProgramContext."""

    def test_all_fields_populated(self) -> None:
        """All fields accessible after construction."""
        ctx = ProgramContext(
            hypothesis_id="h1",
            direction="minimize",
            baseline="val_loss=0.5",
            success_criteria="val_loss < 0.3",
        )
        assert ctx.hypothesis_id == "h1"
        assert ctx.direction == "minimize"
        assert ctx.baseline == "val_loss=0.5"
        assert ctx.success_criteria == "val_loss < 0.3"


class TestInsightsGeneratorProtocol:
    """Protocol satisfaction checks."""

    def test_isinstance_check(self) -> None:
        """OpenAIInsightsGenerator satisfies InsightsGenerator Protocol."""
        gen = OpenAIInsightsGenerator(client=AsyncMock())
        assert isinstance(gen, InsightsGenerator)


# ===================================================================
# LI-12: OpenAIInsightsGenerator adapter tests
# ===================================================================


class TestOpenAIInsightsGeneratorGenerate:
    """Verify generate() call shape and response parsing."""

    @pytest.mark.asyncio
    async def test_generate_sends_correct_prompt(self) -> None:
        """generate() sends hypotheses + program contexts with json_object
        response_format."""
        mock_client = AsyncMock()
        mock_response = _mock_insights_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIInsightsGenerator(client=mock_client, model="gpt-4o-mini")

        hypotheses = _mock_hypothesis_summaries()
        contexts = _mock_program_contexts()
        result = await gen.generate(hypotheses, contexts)

        # Verify call shape
        create_call = mock_client.chat.completions.create
        assert create_call.call_count == 1

        kwargs = create_call.call_args[1]
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["response_format"] == {"type": "json_object"}

        messages = kwargs["messages"]
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert "insights analyst" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"

        # User prompt contains hypothesis data
        user_content = messages[1]["content"]
        assert "Experiment Results" in user_content
        assert "h1" in user_content
        assert "h2" in user_content

        # Program contexts are in the prompt
        assert "Program Contexts" in user_content
        assert "minimize" in user_content
        assert "val_loss=0.5" in user_content

        # Verify return
        assert isinstance(result, InsightsReport)
        assert len(result.patterns) == 1
        assert result.patterns[0].title == "Consistent latency improvement"

    @pytest.mark.asyncio
    async def test_generate_api_exception_returns_none(self) -> None:
        """generate() returns None on API exception (ERR-01)."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = TimeoutError(
            "timeout"
        )

        gen = OpenAIInsightsGenerator(client=mock_client)
        hypotheses = _mock_hypothesis_summaries()
        result = await gen.generate(hypotheses)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_null_content_returns_none(self) -> None:
        """generate() returns None on null message content (ERR-05)."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIInsightsGenerator(client=mock_client)
        hypotheses = _mock_hypothesis_summaries()
        result = await gen.generate(hypotheses)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_malformed_json_returns_none(self) -> None:
        """generate() returns None on invalid JSON (ERR-02)."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "not valid json{{{"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIInsightsGenerator(client=mock_client)
        hypotheses = _mock_hypothesis_summaries()
        result = await gen.generate(hypotheses)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_non_dict_response_returns_none(self) -> None:
        """generate() returns None when response is not a dict (ERR-03)."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "[1, 2, 3]"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIInsightsGenerator(client=mock_client)
        hypotheses = _mock_hypothesis_summaries()
        result = await gen.generate(hypotheses)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_missing_patterns_key_partial_result(self) -> None:
        """Response missing 'patterns' key → InsightsReport with empty
        patterns, populated other sections (ERR-04)."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = json.dumps(
            {
                "recommendations": [
                    {
                        "action": "Do X",
                        "priority": "high",
                        "rationale": "Evidence",
                        "supporting_evidence": ["h1"],
                    }
                ],
                "evidence_correlation": [
                    {
                        "paper_id": "P1",
                        "evidence_summary": "Summary",
                        "related_hypotheses": ["h1"],
                        "code_areas": ["code.py"],
                    }
                ],
            }
        )
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIInsightsGenerator(client=mock_client)
        hypotheses = _mock_hypothesis_summaries()
        result = await gen.generate(hypotheses)

        assert isinstance(result, InsightsReport)
        assert result.patterns == []
        assert len(result.recommendations) == 1
        assert result.recommendations[0].action == "Do X"
        assert len(result.evidence_correlation) == 1
        assert result.evidence_correlation[0].paper_id == "P1"

    @pytest.mark.asyncio
    async def test_generate_default_model_is_gpt4o_mini(self) -> None:
        """Default model is gpt-4o-mini when not specified."""
        gen = OpenAIInsightsGenerator(client=AsyncMock())
        assert gen._model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_generate_empty_hypotheses_makes_llm_call(self) -> None:
        """Empty hypotheses list still makes LLM call (EDGE-01)."""
        mock_client = AsyncMock()
        mock_response = _mock_insights_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIInsightsGenerator(client=mock_client)
        result = await gen.generate([])

        # Call should still be made (code doesn't short-circuit)
        assert mock_client.chat.completions.create.call_count == 1

        # User prompt contains empty hypotheses array
        user_content = (
            mock_client.chat.completions.create.call_args[1]["messages"][
                1
            ]["content"]
        )
        assert "Experiment Results" in user_content

        # Result depends on mock response
        assert isinstance(result, InsightsReport)

    @pytest.mark.asyncio
    async def test_generate_partial_section_parse(self) -> None:
        """Missing recommendations AND evidence_correlation → filled with
        empty lists (not None)."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = json.dumps(
            {
                "patterns": [
                    {
                        "title": "Only pattern",
                        "description": "Just one.",
                        "affected_hypotheses": ["h1"],
                        "confidence": "medium",
                    }
                ],
            }
        )
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIInsightsGenerator(client=mock_client)
        hypotheses = _mock_hypothesis_summaries()
        result = await gen.generate(hypotheses)

        assert isinstance(result, InsightsReport)
        assert len(result.patterns) == 1
        assert result.patterns[0].title == "Only pattern"
        assert result.recommendations == []
        assert result.evidence_correlation == []

    @pytest.mark.asyncio
    async def test_generate_program_contexts_default_none(self) -> None:
        """program_contexts=None is handled (default parameter)."""
        mock_client = AsyncMock()
        mock_response = _mock_insights_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIInsightsGenerator(client=mock_client)
        hypotheses = _mock_hypothesis_summaries()
        result = await gen.generate(hypotheses)  # No program_contexts arg

        assert isinstance(result, InsightsReport)
        assert mock_client.chat.completions.create.call_count == 1


# ===================================================================
# LI-14: Formatter tests — insights rendering
# ===================================================================


class TestFormatMarkdownWithInsights:
    """Markdown rendering with insights section."""

    def test_format_markdown_with_insights(self) -> None:
        """Report with InsightsReport produces 'Cross-Hypothesis Insights'
        section."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=_mock_hypothesis_summaries(),
            insights=_sample_insights_report(),
        )
        md = format_markdown(report)

        assert "## Cross-Hypothesis Insights" in md
        assert "### Patterns" in md
        assert "### Recommendations" in md
        assert "### Evidence Correlations" in md
        assert "Consistent latency improvement" in md
        assert "[high]" in md
        assert "Adopt batching" in md
        assert "Smith et al. 2024" in md

    def test_format_markdown_without_insights(self) -> None:
        """insights=None → NO insights section (backward compat)."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=_mock_hypothesis_summaries(),
            insights=None,
        )
        md = format_markdown(report)

        assert "## Cross-Hypothesis Insights" not in md
        assert "### Patterns" not in md
        # Still has hypotheses
        assert "## Hypothesis: h1" in md

    def test_format_markdown_empty_patterns_section(self) -> None:
        """InsightsReport with empty patterns but populated evidence →
        patterns placeholder shown."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=_mock_hypothesis_summaries(),
            insights=InsightsReport(
                patterns=[],
                recommendations=[],
                evidence_correlation=[
                    EvidenceCorrelation(
                        paper_id="Paper X",
                        evidence_summary="Evidence.",
                        related_hypotheses=["h1"],
                        code_areas=["file.py"],
                    ),
                ],
            ),
        )
        md = format_markdown(report)

        assert "### Patterns" in md
        assert "No patterns detected across hypotheses." in md
        assert "### Recommendations" in md
        assert "No recommendations." in md
        assert "### Evidence Correlations" in md
        assert "Paper X" in md


class TestFormatJsonWithInsights:
    """JSON rendering with insights key."""

    def test_format_json_with_insights(self) -> None:
        """Report with InsightsReport → 'insights' key present."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=_mock_hypothesis_summaries(),
            insights=_sample_insights_report(),
        )
        json_str = format_json(report)
        parsed = json.loads(json_str)

        assert "insights" in parsed
        assert "patterns" in parsed["insights"]
        assert "recommendations" in parsed["insights"]
        assert "evidence_correlation" in parsed["insights"]
        assert len(parsed["insights"]["patterns"]) == 1
        assert (
            parsed["insights"]["patterns"][0]["title"]
            == "Consistent latency improvement"
        )

    def test_format_json_without_insights(self) -> None:
        """insights=None → 'insights' key ABSENT from JSON (backward
        compat)."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=_mock_hypothesis_summaries(),
            insights=None,
        )
        json_str = format_json(report)
        parsed = json.loads(json_str)

        assert "insights" not in parsed
        assert "hypotheses" in parsed

    def test_format_json_empty_recommendations(self) -> None:
        """Empty recommendations list serialized as []."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=_mock_hypothesis_summaries(),
            insights=InsightsReport(
                patterns=[
                    PatternInsight(
                        title="T",
                        description="D",
                        affected_hypotheses=["h1"],
                        confidence="medium",
                    ),
                ],
                recommendations=[],
                evidence_correlation=[],
            ),
        )
        json_str = format_json(report)
        parsed = json.loads(json_str)

        assert parsed["insights"]["recommendations"] == []
        assert parsed["insights"]["evidence_correlation"] == []

    def test_format_json_empty_insights_no_insights_key(self) -> None:
        """Empty InsightsReport (all sections empty) still produces
        'insights' key."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=_mock_hypothesis_summaries(),
            insights=InsightsReport(),
        )
        json_str = format_json(report)
        parsed = json.loads(json_str)

        assert "insights" in parsed
        assert parsed["insights"]["patterns"] == []
        assert parsed["insights"]["recommendations"] == []
        assert parsed["insights"]["evidence_correlation"] == []


# ===================================================================
# LI-15: CLI integration test — --with-insights flag
# ===================================================================


class MockInsightsGenerator:
    """Test double satisfying InsightsGenerator Protocol."""

    def __init__(
        self,
        return_value: InsightsReport | None = None,
        should_raise: bool = False,
    ) -> None:
        self.return_value = return_value
        self.should_raise = should_raise
        self.generate_called = False
        self.last_hypotheses = None
        self.last_contexts = None

    async def generate(
        self,
        hypotheses: list,
        program_contexts: list | None = None,
    ) -> InsightsReport | None:
        self.generate_called = True
        self.last_hypotheses = hypotheses
        self.last_contexts = program_contexts
        if self.should_raise:
            raise RuntimeError("LLM failure")
        return self.return_value


class TestCLIInsightsFlag:
    """CLI --with-insights flag wiring."""

    def test_report_with_insights_flag_activates_generator(self):
        """--with-insights creates OpenAIInsightsGenerator and passes
        to compile_all."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"
            exp_dir.mkdir(parents=True)
            hyp_dir = exp_dir / "h1"
            hyp_dir.mkdir()
            (hyp_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\t"
                "baseline_value\tdelta\tstatus\ttimestamp\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            # program.md
            (hyp_dir / "program.md").write_text(
                "---\nhypothesis_id: h1\ndirection: minimize\n"
                "baseline:\n  val_loss: 0.5\nsuccess_criteria: val_loss < 0.3\n"
                "time_budget: 30m\nmax_iterations: 5\n"
                "run_command: python train.py\n"
                "coding_agent_model: test\n"
                "target_metric: val_loss\n---\n",
                encoding="utf-8",
            )

            mock_insights_report = _sample_insights_report()

            with patch(
                "research_to_dev.cli.main.load_dotenv"
            ) as mock_dotenv, patch.dict(
                os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
            ), patch(
                "research_to_dev.cli.main.AsyncOpenAI"
            ) as mock_ai, patch(
                "research_to_dev.report.insights.AsyncOpenAI"
            ) as mock_insights_ai:
                # Mock the insights generation — replace generate with
                # a coroutine that returns our report
                original_generate = (
                    OpenAIInsightsGenerator.generate
                )

                async def _mock_generate(self, hypotheses, program_contexts=None):
                    return mock_insights_report

                with patch.object(
                    OpenAIInsightsGenerator,
                    "generate",
                    _mock_generate,
                ):
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(tmpdir)
                        runner = CliRunner()
                        result = runner.invoke(
                            app, ["report", "--with-insights"]
                        )

                        assert result.exit_code == 0, (
                            f"Output: {result.output}"
                        )
                        assert "Report written to" in result.output

                        # Read the generated report
                        reports_dir = (
                            Path(tmpdir) / ".research-to-dev" / "reports"
                        )
                        md_files = list(reports_dir.glob("*.md"))
                        assert len(md_files) >= 1
                        md_content = md_files[0].read_text(
                            encoding="utf-8"
                        )
                        assert "Cross-Hypothesis Insights" in md_content
                    finally:
                        os.chdir(original_cwd)

    def test_report_without_insights_flag_no_generator(self):
        """Default behavior: no --with-insights → report has no insights
        section (observable behavior test)."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"
            exp_dir.mkdir(parents=True)
            hyp_dir = exp_dir / "h1"
            hyp_dir.mkdir()
            (hyp_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\t"
                "baseline_value\tdelta\tstatus\ttimestamp\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            (hyp_dir / "program.md").write_text(
                "---\nhypothesis_id: h1\ndirection: minimize\n"
                "baseline:\n  val_loss: 0.5\nsuccess_criteria: val_loss < 0.3\n"
                "time_budget: 30m\nmax_iterations: 5\n"
                "run_command: python train.py\n"
                "coding_agent_model: test\n"
                "target_metric: val_loss\n---\n",
                encoding="utf-8",
            )

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runner = CliRunner()
                result = runner.invoke(app, ["report"])

                assert result.exit_code == 0, (
                    f"Output: {result.output}"
                )

                # Report should NOT have insights section
                reports_dir = (
                    Path(tmpdir) / ".research-to-dev" / "reports"
                )
                md_files = list(reports_dir.glob("*.md"))
                assert len(md_files) >= 1
                md_content = md_files[0].read_text(encoding="utf-8")
                assert "Cross-Hypothesis Insights" not in md_content
            finally:
                os.chdir(original_cwd)

    def test_report_insights_failure_shows_warning(self):
        """LLM call fails → warning printed to stderr, report still
        written with insights=None."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"
            exp_dir.mkdir(parents=True)
            hyp_dir = exp_dir / "h1"
            hyp_dir.mkdir()
            (hyp_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\t"
                "baseline_value\tdelta\tstatus\ttimestamp\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            (hyp_dir / "program.md").write_text(
                "---\nhypothesis_id: h1\ndirection: minimize\n"
                "baseline:\n  val_loss: 0.5\nsuccess_criteria: val_loss < 0.3\n"
                "time_budget: 30m\nmax_iterations: 5\n"
                "run_command: python train.py\n"
                "coding_agent_model: test\n"
                "target_metric: val_loss\n---\n",
                encoding="utf-8",
            )

            async def _mock_generate_fail(self, hypotheses, program_contexts=None):
                raise RuntimeError("LLM API timeout")

            with patch(
                "research_to_dev.cli.main.load_dotenv"
            ) as mock_dotenv, patch.dict(
                os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False
            ), patch(
                "research_to_dev.cli.main.AsyncOpenAI"
            ) as mock_ai, patch.object(
                OpenAIInsightsGenerator,
                "generate",
                _mock_generate_fail,
            ):
                original_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    runner = CliRunner()
                    result = runner.invoke(
                        app, ["report", "--with-insights"]
                    )

                    # Should still succeed (graceful degradation)
                    assert result.exit_code == 0, (
                        f"Output: {result.output}"
                    )
                    assert "Report written to" in result.output

                    # Warning should be printed to stderr
                    assert (
                        "Warning: Insights generation failed" in result.output
                    )

                    # Report should NOT have insights section
                    reports_dir = (
                        Path(tmpdir) / ".research-to-dev" / "reports"
                    )
                    md_files = list(reports_dir.glob("*.md"))
                    md_content = md_files[0].read_text(encoding="utf-8")
                    assert "Cross-Hypothesis Insights" not in md_content
                finally:
                    os.chdir(original_cwd)

    def test_report_insights_no_api_key_warns(self):
        """Missing OPENAI_API_KEY → warning, no crash."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"
            exp_dir.mkdir(parents=True)
            hyp_dir = exp_dir / "h1"
            hyp_dir.mkdir()
            (hyp_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\t"
                "baseline_value\tdelta\tstatus\ttimestamp\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            (hyp_dir / "program.md").write_text(
                "---\nhypothesis_id: h1\ndirection: minimize\n"
                "baseline:\n  val_loss: 0.5\nsuccess_criteria: val_loss < 0.3\n"
                "time_budget: 30m\nmax_iterations: 5\n"
                "run_command: python train.py\n"
                "coding_agent_model: test\n"
                "target_metric: val_loss\n---\n",
                encoding="utf-8",
            )

            with patch(
                "research_to_dev.cli.main.load_dotenv"
            ):
                original_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    runner = CliRunner()
                    result = runner.invoke(
                        app,
                        ["report", "--with-insights"],
                        env={"OPENAI_API_KEY": ""},
                    )

                    # Should succeed without insights
                    assert result.exit_code == 0, (
                        f"Output: {result.output}"
                    )
                    assert (
                        "OPENAI_API_KEY not set" in result.output
                    )
                    assert "Report written to" in result.output
                finally:
                    os.chdir(original_cwd)
