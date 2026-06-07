"""Tests for format_markdown and format_json — output structure,
content verification, and edge cases per RC-14."""

from __future__ import annotations

import json

from research_to_dev.report.compilation import (
    BaselineComparison,
    CompiledReport,
    HypothesisSummary,
    IterationResult,
    format_json,
    format_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_iterations() -> list[IterationResult]:
    """Return a representative set of iterations."""
    return [
        IterationResult(
            iteration=1,
            metric_name="val_loss",
            metric_value=0.45,
            baseline_value=0.50,
            delta=-0.05,
            status="success",
            timestamp="2026-06-07T14:00:00Z",
        ),
        IterationResult(
            iteration=2,
            metric_name="val_loss",
            metric_value=0.42,
            baseline_value=0.45,
            delta=-0.03,
            status="success",
            timestamp="2026-06-07T14:05:00Z",
        ),
        IterationResult(
            iteration=3,
            metric_name="val_loss",
            metric_value=0.48,
            baseline_value=0.42,
            delta=0.06,
            status="failed",
            timestamp="2026-06-07T14:10:00Z",
        ),
    ]


def _sample_report() -> CompiledReport:
    """Return a CompiledReport with 2 hypotheses."""
    return CompiledReport(
        generated_at="2026-06-07T14:30:00Z",
        hypotheses=[
            HypothesisSummary(
                hypothesis_id="abc123",
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
                iterations=_sample_iterations(),
            ),
            HypothesisSummary(
                hypothesis_id="def456",
                status="crash",
                iterations_total=2,
                iterations_kept=0,
                iterations_discarded=0,
                iterations_crashed=2,
                best_metric_value=None,
                baseline_comparison=None,
                iterations=[
                    IterationResult(
                        iteration=1,
                        metric_name="accuracy",
                        metric_value=0.0,
                        baseline_value=0.90,
                        delta=-0.90,
                        status="failed",
                        timestamp="2026-06-07T14:30:00Z",
                    ),
                    IterationResult(
                        iteration=2,
                        metric_name="accuracy",
                        metric_value=0.0,
                        baseline_value=0.90,
                        delta=-0.90,
                        status="timeout",
                        timestamp="2026-06-07T14:35:00Z",
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# RC-14: format_markdown tests
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    """Markdown formatting structure and content."""

    def test_contains_header_with_timestamp(self) -> None:
        """Markdown output starts with # header + timestamp."""
        report = _sample_report()
        md = format_markdown(report)

        assert md.startswith("# Experiment Results — 2026-06-07T14:30:00Z")

    def test_contains_hypothesis_sections(self) -> None:
        """Each hypothesis gets its own ## section."""
        md = format_markdown(_sample_report())

        assert "## Hypothesis: abc123" in md
        assert "## Hypothesis: def456" in md

    def test_contains_status_field(self) -> None:
        """Status is shown for each hypothesis."""
        md = format_markdown(_sample_report())

        assert "- **Status**: improved" in md
        assert "- **Status**: crash" in md

    def test_contains_iteration_counts(self) -> None:
        """Total/kept/discarded/crashed counts are shown."""
        md = format_markdown(_sample_report())

        assert "3 total" in md
        assert "2 kept" in md
        assert "1 crashed" in md

    def test_contains_best_metric(self) -> None:
        """Best metric value and delta are shown."""
        md = format_markdown(_sample_report())

        assert "val_loss=0.42" in md
        # best_metric_value is 0.42 (from iteration 2), delta is -0.03
        assert "(delta: -0.0300)" in md

    def test_contains_baseline_comparison(self) -> None:
        """Baseline comparison with trend and best improvement."""
        md = format_markdown(_sample_report())

        assert "- **Baseline comparison**: improved (best improvement: -0.0500)" in md

    def test_crash_hypothesis_shows_na(self) -> None:
        """Crash hypothesis shows N/A for metrics."""
        md = format_markdown(_sample_report())

        assert "- **Best metric**: N/A" in md
        assert "N/A (no successful iterations)" in md

    def test_contains_iteration_table(self) -> None:
        """Iteration table with header and data rows."""
        md = format_markdown(_sample_report())

        assert "| # | Metric | Value | Delta | Status |" in md
        assert "| 1 | val_loss | 0.45 | -0.0500 | success |" in md
        assert "| 3 | val_loss | 0.48 | +0.0600 | failed |" in md

    def test_empty_report_no_iterations(self) -> None:
        """Empty report shows appropriate message."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=[],
        )
        md = format_markdown(report)

        assert "*No hypothesis results to report.*" in md

    def test_no_iterations_hypothesis_graceful(self) -> None:
        """Hypothesis with no iterations shows appropriate message."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=[
                HypothesisSummary(
                    hypothesis_id="xyz",
                    status="no_iterations",
                    iterations_total=0,
                    iterations_kept=0,
                    iterations_discarded=0,
                    iterations_crashed=0,
                    best_metric_value=None,
                    baseline_comparison=None,
                    iterations=[],
                ),
            ],
        )
        md = format_markdown(report)

        assert "## Hypothesis: xyz" in md
        assert "- **Status**: no_iterations" in md
        assert "*No iterations recorded.*" in md


# ---------------------------------------------------------------------------
# RC-14: format_json tests
# ---------------------------------------------------------------------------


class TestFormatJson:
    """JSON formatting structure and content."""

    def test_valid_json_output(self) -> None:
        """Output is valid JSON and can be parsed back."""
        report = _sample_report()
        json_str = format_json(report)

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_contains_generated_at(self) -> None:
        """JSON includes generated_at field."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        assert parsed["generated_at"] == "2026-06-07T14:30:00Z"

    def test_contains_hypotheses_array(self) -> None:
        """JSON includes hypotheses array with correct count."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        assert isinstance(parsed["hypotheses"], list)
        assert len(parsed["hypotheses"]) == 2

    def test_hypothesis_fields_present(self) -> None:
        """Each hypothesis has all required fields."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        h = parsed["hypotheses"][0]
        assert "hypothesis_id" in h
        assert "status" in h
        assert "iterations_total" in h
        assert "iterations_kept" in h
        assert "iterations_discarded" in h
        assert "iterations_crashed" in h
        assert "best_metric_value" in h
        assert "baseline_comparison" in h
        assert "iterations" in h

    def test_iteration_fields_present(self) -> None:
        """Each iteration in JSON has all fields."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        it = parsed["hypotheses"][0]["iterations"][0]
        assert "iteration" in it
        assert "metric_name" in it
        assert "metric_value" in it
        assert "baseline_value" in it
        assert "delta" in it
        assert "status" in it
        assert "timestamp" in it

    def test_null_baseline_comparison_serialized(self) -> None:
        """Null baseline_comparison is serialized as null."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        crash_h = parsed["hypotheses"][1]  # def456 is crash
        assert crash_h["baseline_comparison"] is None

    def test_json_roundtrip_preserves_data(self) -> None:
        """JSON round-trip preserves all hypothesis data."""
        report = _sample_report()
        json_str = format_json(report)
        parsed = json.loads(json_str)

        h0 = parsed["hypotheses"][0]
        assert h0["hypothesis_id"] == "abc123"
        assert h0["status"] == "improved"
        assert h0["iterations_total"] == 3
        assert h0["iterations_kept"] == 2
        assert h0["best_metric_value"] == 0.42
        assert h0["baseline_comparison"]["overall_trend"] == "improved"
        assert h0["baseline_comparison"]["best_improvement"] == -0.05

    def test_empty_report_json(self) -> None:
        """Empty report produces valid JSON with empty hypotheses."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=[],
        )
        json_str = format_json(report)
        parsed = json.loads(json_str)

        assert parsed["hypotheses"] == []
        assert parsed["generated_at"] == "2026-06-07T14:30:00Z"

    def test_json_indented_output(self) -> None:
        """JSON output is indented for readability."""
        json_str = format_json(_sample_report())
        assert "  " in json_str  # indent=2
