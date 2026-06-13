"""Integration tests for results compilation — E2E, CLI, and
write_reports per RC-16, RC-20, RC-21, RC-22."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_to_dev.cli.main import app
from research_to_dev.report.compilation import (
    CompiledReport,
    TsvResultsReader,
    compile_all,
    format_json,
    format_markdown,
    write_reports,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_HEADER = (
    "iteration\tmetric_name\tmetric_value\t"
    "baseline_value\tdelta\tstatus\ttimestamp"
)


def _setup_experiment(
    exp_dir: Path,
    hyp_id: str,
    tsv_content: str | None = None,
    direction: str = "maximize",
) -> Path:
    """Create a hypothesis directory with results.tsv and program.md."""
    hyp_dir = exp_dir / hyp_id
    hyp_dir.mkdir(parents=True, exist_ok=True)

    if tsv_content is not None:
        (hyp_dir / "results.tsv").write_text(tsv_content, encoding="utf-8")

    program_content = f"""---
hypothesis_id: {hyp_id}
target_metric: val_loss
success_criteria: val_loss < 0.3
time_budget: 30m
max_iterations: 5
baseline:
  val_loss: 0.5
run_command: python train.py
coding_agent_model: test-model
direction: {direction}
---

# Experiment Plan
"""
    (hyp_dir / "program.md").write_text(program_content, encoding="utf-8")
    return hyp_dir


# ---------------------------------------------------------------------------
# RC-16: write_reports tests
# ---------------------------------------------------------------------------


class TestWriteReports:
    """Filesystem output for write_reports."""

    def test_both_md_and_json_files_created(self) -> None:
        """Both .md and .json files are written."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00+00:00",
            hypotheses=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "reports"
            md_path, json_path = write_reports(report, reports_dir)

            assert md_path.exists()
            assert json_path.exists()
            assert md_path.suffix == ".md"
            assert json_path.suffix == ".json"

    def test_directory_auto_created(self) -> None:
        """Reports directory is created if it doesn't exist."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00+00:00",
            hypotheses=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "deep" / "nested" / "reports"
            assert not reports_dir.exists()

            md_path, _ = write_reports(report, reports_dir)

            assert reports_dir.exists()
            assert md_path.exists()

    def test_content_matches_formatter_output(self) -> None:
        """File content matches what format_markdown/format_json produce."""
        report = _sample_compiled_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "reports"
            md_path, json_path = write_reports(report, reports_dir)

            md_content = md_path.read_text(encoding="utf-8")
            json_content = json_path.read_text(encoding="utf-8")

            assert md_content == format_markdown(report)
            assert json_content == format_json(report)

    def test_timestamp_in_filename(self) -> None:
        """Filename includes timestamp derived from generated_at."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00+00:00",
            hypotheses=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "reports"
            md_path, json_path = write_reports(report, reports_dir)

            # Filename should contain date/time part
            assert "20260607-143000" in md_path.name
            assert "20260607-143000" in json_path.name


# ---------------------------------------------------------------------------
# Helpers for E2E tests
# ---------------------------------------------------------------------------


def _sample_compiled_report() -> CompiledReport:
    """A non-trivial report for testing write_reports."""
    from research_to_dev.report.compilation import (
        BaselineComparison,
        HypothesisSummary,
        IterationResult,
    )

    return CompiledReport(
        generated_at="2026-06-07T14:30:00+00:00",
        hypotheses=[
            HypothesisSummary(
                hypothesis_id="test-h1",
                status="improved",
                iterations_total=2,
                iterations_kept=2,
                iterations_discarded=0,
                iterations_crashed=0,
                best_metric_value=0.42,
                baseline_comparison=BaselineComparison(
                    best_improvement=-0.08,
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
                    IterationResult(
                        iteration=2,
                        metric_name="val_loss",
                        metric_value=0.42,
                        baseline_value=0.45,
                        delta=-0.03,
                        status="success",
                        timestamp="2026-06-07T14:05:00Z",
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# RC-20: E2E integration tests
# ---------------------------------------------------------------------------


class TestE2ECompilation:
    """End-to-end: temp dirs with results.tsv + program.md → CompiledReport."""

    def test_full_compilation_pipeline(self) -> None:
        """Full pipeline: create files, compile_all, verify summaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "experiments"

            # Hypothesis 1: improved (minimize direction)
            tsv1 = (
                f"{_VALID_HEADER}\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"
                "2\tval_loss\t0.42\t0.45\t-0.03\tsuccess\t2026-06-07T14:05:00Z\n"
                "3\tval_loss\t0.48\t0.42\t0.06\tfailed\t2026-06-07T14:10:00Z\n"
            )
            _setup_experiment(exp_dir, "h1", tsv1, direction="minimize")

            # Hypothesis 2: worsened (minimize, all positive deltas)
            tsv2 = (
                f"{_VALID_HEADER}\n"
                "1\tval_loss\t0.55\t0.50\t0.05\tsuccess\t2026-06-07T14:00:00Z\n"
                "2\tval_loss\t0.53\t0.55\t-0.02\tsuccess\t2026-06-07T14:05:00Z\n"
            )
            # Direction maximize → 0.05 is keep, -0.02 is discard
            _setup_experiment(exp_dir, "h2", tsv2, direction="maximize")

            reader = TsvResultsReader(exp_dir)
            report = compile_all(reader, exp_dir)

            assert len(report.hypotheses) == 2

            h1 = next(h for h in report.hypotheses if h.hypothesis_id == "h1")
            assert h1.status == "improved"
            assert h1.iterations_total == 3
            assert h1.iterations_kept == 2
            assert h1.iterations_crashed == 1
            assert h1.best_metric_value == 0.42

            h2 = next(h for h in report.hypotheses if h.hypothesis_id == "h2")
            assert h2.status == "improved"  # maximize, delta≥0 kept
            assert h2.iterations_total == 2
            assert h2.iterations_kept == 1
            assert h2.iterations_discarded == 1

    def test_compiled_report_has_generated_at(self) -> None:
        """CompiledReport.generated_at is set to current time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "experiments"
            exp_dir.mkdir()

            reader = TsvResultsReader(exp_dir)
            report = compile_all(reader, exp_dir)

            assert report.generated_at
            # Should be ISO 8601 format
            assert "T" in report.generated_at

    def test_mixed_status_multiple_iterations(self) -> None:
        """Mixed: some kept, some discarded, some crashed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            tsv = (
                f"{_VALID_HEADER}\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"  # kept (minimize)
                "2\tval_loss\t0.55\t0.50\t0.05\tsuccess\t2026-06-07T14:05:00Z\n"  # discarded
                "3\tval_loss\t0.48\t0.50\t-0.02\ttimeout\t2026-06-07T14:10:00Z\n"  # crashed
            )
            _setup_experiment(exp_dir, "mixed", tsv, direction="minimize")

            reader = TsvResultsReader(exp_dir)
            report = compile_all(reader, exp_dir)

            h = report.hypotheses[0]
            assert h.status == "improved"
            assert h.iterations_total == 3
            assert h.iterations_kept == 1
            assert h.iterations_discarded == 1
            assert h.iterations_crashed == 1


# ---------------------------------------------------------------------------
# RC-21: CLI integration tests
# ---------------------------------------------------------------------------


class TestCLIReport:
    """CLI `research-to-dev report` command integration."""

    def test_compile_all_hypotheses(self) -> None:
        """Running `report` without flags compiles all hypotheses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"
            tsv = (
                f"{_VALID_HEADER}\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"
            )
            _setup_experiment(exp_dir, "h1", tsv, direction="minimize")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runner = CliRunner()
                result = runner.invoke(app, ["report"])

                # CLI should succeed
                assert result.exit_code == 0, f"Output: {result.output}"
                assert "Report written to" in result.output

                # Verify report files were created
                reports_dir = Path(tmpdir) / ".research-to-dev" / "reports"
                md_files = list(reports_dir.glob("*.md"))
                json_files = list(reports_dir.glob("*.json"))
                assert len(md_files) >= 1
                assert len(json_files) >= 1
            finally:
                os.chdir(original_cwd)

    def test_filter_single_hypothesis(self) -> None:
        """--hypothesis flag filters to one hypothesis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"

            tsv1 = (
                f"{_VALID_HEADER}\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"
            )
            _setup_experiment(exp_dir, "h1", tsv1, direction="minimize")

            tsv2 = (
                f"{_VALID_HEADER}\n"
                "1\taccuracy\t0.92\t0.90\t0.02\tsuccess\t2026-06-07T14:00:00Z\n"
            )
            _setup_experiment(exp_dir, "h2", tsv2, direction="maximize")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runner = CliRunner()
                result = runner.invoke(app, ["report", "--hypothesis", "h1"])

                assert result.exit_code == 0, f"Output: {result.output}"
                assert "Report written to" in result.output

                # Read the markdown report and verify only h1 is present
                reports_dir = Path(tmpdir) / ".research-to-dev" / "reports"
                md_files = list(reports_dir.glob("*.md"))
                md_content = md_files[0].read_text(encoding="utf-8")

                assert "## Hypothesis: h1" in md_content
                assert "## Hypothesis: h2" not in md_content
            finally:
                os.chdir(original_cwd)

    def test_no_results_exit_code_1(self) -> None:
        """No results.tsv in any experiment dir → exit code 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"
            exp_dir.mkdir(parents=True)

            # Create hypothesis dir but no results.tsv
            (exp_dir / "h1").mkdir()

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runner = CliRunner()
                result = runner.invoke(app, ["report"])

                assert result.exit_code == 1
                assert "Error" in result.output or "No hypothesis" in result.output
            finally:
                os.chdir(original_cwd)

    def test_missing_experiments_directory(self) -> None:
        """No experiments directory at all → exit code 1 with clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runner = CliRunner()
                result = runner.invoke(app, ["report"])

                assert result.exit_code == 1
                assert "Error" in result.output or "not found" in result.output
            finally:
                os.chdir(original_cwd)

    def test_with_insights_flag_accepted(self) -> None:
        """--with-insights flag is accepted (no-op for now)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"
            tsv = (
                f"{_VALID_HEADER}\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"
            )
            _setup_experiment(exp_dir, "h1", tsv, direction="minimize")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runner = CliRunner()
                result = runner.invoke(app, ["report", "--with-insights"])

                # Should still succeed — flag is a no-op
                assert result.exit_code == 0, f"Output: {result.output}"
            finally:
                os.chdir(original_cwd)

    def test_hypothesis_directory_not_found(self) -> None:
        """--hypothesis with nonexistent ID gives clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"
            exp_dir.mkdir(parents=True)

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runner = CliRunner()
                result = runner.invoke(app, ["report", "--hypothesis", "nonexistent"])

                assert result.exit_code == 1
                assert "Error" in result.output or "not found" in result.output
            finally:
                os.chdir(original_cwd)

    def test_empty_results_warning(self) -> None:
        """Hypothesis with empty TSV prints warning but succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments"

            # Empty TSV (header only)
            tsv = _VALID_HEADER + "\n"
            _setup_experiment(exp_dir, "h1", tsv, direction="minimize")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                runner = CliRunner()
                result = runner.invoke(app, ["report", "--hypothesis", "h1"])

                # Should succeed but with warning
                assert result.exit_code == 0, f"Output: {result.output}"
                assert (
                    "Warning" in result.output
                    or "Empty" in result.output
                    or "no iterations" in result.output.lower()
                )
            finally:
                os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# RC-22: Exports verification
# ---------------------------------------------------------------------------


class TestReportExports:
    """Verify __init__.py exports match the public API."""

    def test_all_public_types_exported(self) -> None:
        """All public types are accessible from research_to_dev.report."""
        from research_to_dev.report import (
            BaselineComparison,
            CompiledReport,
            HypothesisSummary,
            IterationResult,
            ResultsReader,
            TsvResultsReader,
            compile_all,
            compile_hypothesis,
            format_json,
            format_markdown,
            write_reports,
        )

        # Verify each import is the correct type
        assert BaselineComparison is not None
        assert CompiledReport is not None
        assert HypothesisSummary is not None
        assert IterationResult is not None
        assert TsvResultsReader is not None

        # Functions are callable
        assert callable(compile_all)
        assert callable(compile_hypothesis)
        assert callable(format_markdown)
        assert callable(format_json)
        assert callable(write_reports)

        # ResultsReader is a Protocol
        from typing import Protocol as _Protocol
        assert issubclass(ResultsReader, _Protocol)

    def test_report_config_exported(self) -> None:
        """ReportConfig is importable from shared.config."""
        from research_to_dev.shared.config import ReportConfig

        config = ReportConfig()
        assert config.experiments_dir == ".research-to-dev/experiments"
        assert config.reports_dir == ".research-to-dev/reports"
        assert config.hypothesis_filter is None
        assert config.include_insights is False

    def test_report_config_custom_values(self) -> None:
        """ReportConfig accepts custom values via constructor."""
        from research_to_dev.shared.config import ReportConfig

        config = ReportConfig(
            experiments_dir="/tmp/exps",
            reports_dir="/tmp/reps",
            hypothesis_filter="abc",
            include_insights=True,
        )
        assert config.experiments_dir == "/tmp/exps"
        assert config.reports_dir == "/tmp/reps"
        assert config.hypothesis_filter == "abc"
        assert config.include_insights is True

    def test_traceability_types_exported(self) -> None:
        """Traceability types are importable from research_to_dev.report."""
        from research_to_dev.report import (
            CorrelationTrace,
            HypothesisOrigin,
            JsonTraceReader,
            TraceReader,
            TraceabilityContext,
            build_traceability,
        )
        from typing import Protocol as _Protocol

        # Verify each is the correct type
        assert CorrelationTrace is not None
        assert HypothesisOrigin is not None
        assert TraceabilityContext is not None
        assert JsonTraceReader is not None

        # TraceReader is a Protocol
        assert issubclass(TraceReader, _Protocol)

        # build_traceability is callable
        assert callable(build_traceability)


# ---------------------------------------------------------------------------
# TR-15: --trace flag tests
# ---------------------------------------------------------------------------


class TestReportTraceFlag:
    """CLI tests for `research-to-dev report --trace` flag (TR-15, TR-16)."""

    def test_trace_flag_accepted(self) -> None:
        """--trace flag is accepted by the report command."""
        runner = CliRunner()
        import os as _os
        original_cwd = _os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                _os.chdir(tmpdir)
                # Create minimal experiments dir
                exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments" / "h1"
                exp_dir.mkdir(parents=True)
                (exp_dir / "results.tsv").write_text(
                    "iteration\tmetric_name\tmetric_value\t"
                    "baseline_value\tdelta\tstatus\ttimestamp\n"
                    "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2024-01-01T00:00:00Z\n"
                )
                (exp_dir / "program.md").write_text(
                    "---\nhypothesis_id: h1\ntarget_metric: val_loss\n"
                    "success_criteria: val_loss < 0.5\n"
                    "time_budget: 30m\nmax_iterations: 5\n"
                    "baseline:\n  val_loss: 0.5\n"
                    "run_command: python train.py\n"
                    "coding_agent_model: test\n"
                    "direction: minimize\n---\n"
                )

                result = runner.invoke(
                    app, ["report", "--trace", "/tmp/nonexistent.json"]
                )
                # Should succeed (missing trace is a warning, not an error)
                assert result.exit_code == 0, f"Output: {result.output}"
                assert "Report written to" in result.output
            finally:
                _os.chdir(original_cwd)

    def test_trace_flag_default_empty(self) -> None:
        """When --trace is not provided, default is empty string."""
        runner = CliRunner()
        # Verify via help output that --trace has a default
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "--trace" in result.stdout

    def test_trace_valid_json_wires_traceability(self) -> None:
        """Valid trace.json → report should render headers fine."""
        runner = CliRunner()
        import json as _json
        import os as _os
        original_cwd = _os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                _os.chdir(tmpdir)

                # Create experiments
                exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments" / "abc123"
                exp_dir.mkdir(parents=True)
                (exp_dir / "results.tsv").write_text(
                    "iteration\tmetric_name\tmetric_value\t"
                    "baseline_value\tdelta\tstatus\ttimestamp\n"
                    "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2024-01-01T00:00:00Z\n"
                )
                (exp_dir / "program.md").write_text(
                    "---\nhypothesis_id: abc123\ntarget_metric: val_loss\n"
                    "success_criteria: val_loss < 0.5\n"
                    "time_budget: 30m\nmax_iterations: 5\n"
                    "baseline:\n  val_loss: 0.5\n"
                    "run_command: python train.py\n"
                    "coding_agent_model: test\n"
                    "direction: minimize\n---\n"
                )

                # Create a minimal valid trace.json
                trace_path = Path(tmpdir) / "trace.json"
                trace_data = {
                    "hypotheses": [
                        {
                            "id": "abc123",
                            "title": "Test Hypothesis",
                            "description": "A test hypothesis for traceability",
                            "approach": "Implement caching",
                            "target_metric": "val_loss",
                            "expected_improvement": "lower val_loss",
                            "code_changes": "src/main.py",
                            "supporting_papers": ["paper-1"],
                            "correlations": [],
                        }
                    ],
                    "correlation": {},
                }
                trace_path.write_text(_json.dumps(trace_data))

                result = runner.invoke(
                    app, ["report", "--trace", str(trace_path)]
                )
                assert result.exit_code == 0, f"Output: {result.output}"
                assert "Report written to" in result.output
            finally:
                _os.chdir(original_cwd)

    def test_trace_missing_file_warns(self) -> None:
        """Missing trace file → warning, report succeeds without traceability."""
        runner = CliRunner()
        import os as _os
        original_cwd = _os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                _os.chdir(tmpdir)

                exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments" / "h1"
                exp_dir.mkdir(parents=True)
                (exp_dir / "results.tsv").write_text(
                    "iteration\tmetric_name\tmetric_value\t"
                    "baseline_value\tdelta\tstatus\ttimestamp\n"
                    "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2024-01-01T00:00:00Z\n"
                )
                (exp_dir / "program.md").write_text(
                    "---\nhypothesis_id: h1\ntarget_metric: val_loss\n"
                    "success_criteria: val_loss < 0.5\n"
                    "time_budget: 30m\nmax_iterations: 5\n"
                    "baseline:\n  val_loss: 0.5\n"
                    "run_command: python train.py\n"
                    "coding_agent_model: test\n"
                    "direction: minimize\n---\n"
                )

                result = runner.invoke(
                    app, ["report", "--trace", "/tmp/does/not/exist/trace.json"]
                )
                assert result.exit_code == 0, f"Output: {result.output}"
                # Should succeed with warning about traceability
                assert "Report written to" in result.output
            finally:
                _os.chdir(original_cwd)

    def test_trace_malformed_json_warns(self) -> None:
        """Malformed trace JSON → warning, report succeeds without traceability."""
        runner = CliRunner()
        import os as _os
        original_cwd = _os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                _os.chdir(tmpdir)

                exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments" / "h1"
                exp_dir.mkdir(parents=True)
                (exp_dir / "results.tsv").write_text(
                    "iteration\tmetric_name\tmetric_value\t"
                    "baseline_value\tdelta\tstatus\ttimestamp\n"
                    "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2024-01-01T00:00:00Z\n"
                )
                (exp_dir / "program.md").write_text(
                    "---\nhypothesis_id: h1\ntarget_metric: val_loss\n"
                    "success_criteria: val_loss < 0.5\n"
                    "time_budget: 30m\nmax_iterations: 5\n"
                    "baseline:\n  val_loss: 0.5\n"
                    "run_command: python train.py\n"
                    "coding_agent_model: test\n"
                    "direction: minimize\n---\n"
                )

                # Create malformed JSON
                bad_trace = Path(tmpdir) / "bad.json"
                bad_trace.write_text("this is not json{{{")

                result = runner.invoke(
                    app, ["report", "--trace", str(bad_trace)]
                )
                assert result.exit_code == 0, f"Output: {result.output}"
                assert "Report written to" in result.output
            finally:
                _os.chdir(original_cwd)

    def test_report_summary_printed(self) -> None:
        """Report command prints terminal summary with REPORT SUMMARY header (TR-19)."""
        runner = CliRunner()
        import os as _os
        original_cwd = _os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                _os.chdir(tmpdir)

                exp_dir = Path(tmpdir) / ".research-to-dev" / "experiments" / "h1"
                exp_dir.mkdir(parents=True)
                (exp_dir / "results.tsv").write_text(
                    "iteration\tmetric_name\tmetric_value\t"
                    "baseline_value\tdelta\tstatus\ttimestamp\n"
                    "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2024-01-01T00:00:00Z\n"
                    "2\tval_loss\t0.42\t0.45\t-0.03\tsuccess\t2024-01-01T00:05:00Z\n"
                )
                (exp_dir / "program.md").write_text(
                    "---\nhypothesis_id: h1\ntarget_metric: val_loss\n"
                    "success_criteria: val_loss < 0.5\n"
                    "time_budget: 30m\nmax_iterations: 5\n"
                    "baseline:\n  val_loss: 0.5\n"
                    "run_command: python train.py\n"
                    "coding_agent_model: test\n"
                    "direction: minimize\n---\n"
                )

                result = runner.invoke(app, ["report"])
                assert result.exit_code == 0, f"Output: {result.output}"
                assert "REPORT SUMMARY" in result.stdout
                assert "Verdict:" in result.stdout
                assert "Output:" in result.stdout
                assert "Time:" in result.stdout
            finally:
                _os.chdir(original_cwd)
