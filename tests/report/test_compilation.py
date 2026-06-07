"""Tests for compile_hypothesis and compile_all — compilation logic,
status classification, baseline comparison, and experiment directory
discovery per RC-10 and RC-11."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from research_to_dev.report.compilation import (
    BaselineComparison,
    CompiledReport,
    HypothesisSummary,
    IterationResult,
    TsvResultsReader,
    compile_all,
    compile_hypothesis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mkiter(
    iteration: int,
    metric_value: float = 0.5,
    delta: float = 0.0,
    status: str = "success",
    metric_name: str = "val_loss",
) -> IterationResult:
    """Factory for IterationResult with sensible defaults."""
    return IterationResult(
        iteration=iteration,
        metric_name=metric_name,
        metric_value=metric_value,
        baseline_value=0.5,
        delta=delta,
        status=status,
        timestamp="2026-06-07T14:00:00Z",
    )


def _write_program_md(hyp_dir: Path, direction: str = "maximize") -> None:
    """Write a minimal program.md for direction discovery."""
    content = f"""---
hypothesis_id: {hyp_dir.name}
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
    (hyp_dir / "program.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# RC-10: compile_hypothesis tests
# ---------------------------------------------------------------------------


class TestCompileHypothesisStatus:
    """Status classification: improved, worsened, crash, no_iterations."""

    def test_improved_with_kept_and_discarded(self) -> None:
        """Some success rows kept → status 'improved'."""
        iterations = [
            _mkiter(1, delta=-0.05, metric_value=0.45),  # kept (minimize)
            _mkiter(2, delta=0.03, metric_value=0.53),  # discarded
            _mkiter(3, delta=-0.08, metric_value=0.42),  # kept
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.status == "improved"
        assert summary.iterations_total == 3
        assert summary.iterations_kept == 2
        assert summary.iterations_discarded == 1
        assert summary.iterations_crashed == 0

    def test_worsened_all_discarded(self) -> None:
        """All success rows discarded → status 'worsened'."""
        iterations = [
            _mkiter(1, delta=0.05, metric_value=0.55),  # discarded (minimize)
            _mkiter(2, delta=0.03, metric_value=0.53),  # discarded
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.status == "worsened"
        assert summary.iterations_kept == 0
        assert summary.iterations_discarded == 2

    def test_crash_all_failed_or_timeout(self) -> None:
        """All iterations crashed → status 'crash'."""
        iterations = [
            _mkiter(1, status="failed"),
            _mkiter(2, status="timeout"),
            _mkiter(3, status="failed"),
        ]
        summary = compile_hypothesis("h1", iterations, "maximize")

        assert summary.status == "crash"
        assert summary.iterations_total == 3
        assert summary.iterations_crashed == 3
        assert summary.iterations_kept == 0
        assert summary.iterations_discarded == 0
        assert summary.best_metric_value is None
        assert summary.baseline_comparison is None

    def test_no_iterations_empty_list(self) -> None:
        """Zero iterations → status 'no_iterations', null metrics."""
        summary = compile_hypothesis("h1", [], "maximize")

        assert summary.status == "no_iterations"
        assert summary.iterations_total == 0
        assert summary.iterations_kept == 0
        assert summary.iterations_discarded == 0
        assert summary.iterations_crashed == 0
        assert summary.best_metric_value is None
        assert summary.baseline_comparison is None
        assert summary.iterations == []

    def test_improved_maximize_direction(self) -> None:
        """Maximize direction: positive deltas are kept."""
        iterations = [
            _mkiter(1, delta=0.02, metric_value=0.52),  # kept
            _mkiter(2, delta=0.05, metric_value=0.55),  # kept
        ]
        summary = compile_hypothesis("h1", iterations, "maximize")

        assert summary.status == "improved"
        assert summary.iterations_kept == 2
        assert summary.iterations_discarded == 0

    def test_worsened_minimize_direction(self) -> None:
        """Minimize direction: positive deltas are discarded."""
        iterations = [
            _mkiter(1, delta=0.05, metric_value=0.55),  # discarded
            _mkiter(2, delta=0.02, metric_value=0.52),  # discarded
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.status == "worsened"
        assert summary.iterations_kept == 0
        assert summary.iterations_discarded == 2

    def test_mixed_crashed_and_success(self) -> None:
        """Some crashed + some success kept → still 'improved'."""
        iterations = [
            _mkiter(1, delta=-0.05, metric_value=0.45),  # kept
            _mkiter(2, status="failed"),
            _mkiter(3, status="timeout"),
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.status == "improved"
        assert summary.iterations_total == 3
        assert summary.iterations_kept == 1
        assert summary.iterations_crashed == 2

    def test_zero_kept_all_success_but_discarded(self) -> None:
        """All successes but none favourable → worsened, not crash."""
        iterations = [
            _mkiter(1, delta=0.05, metric_value=0.55),  # all discarded
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.status == "worsened"
        # best_metric_value should still exist (there are successful rows)
        assert summary.best_metric_value is not None
        # baseline_comparison should exist
        assert summary.baseline_comparison is not None
        assert summary.baseline_comparison.overall_trend == "worsened"


class TestCompileHypothesisBaselineComparison:
    """BaselineComparison: best_improvement, overall_trend."""

    def test_best_improvement_maximize_direction(self) -> None:
        """Maximize: best_improvement = max(delta)."""
        iterations = [
            _mkiter(1, delta=0.02, metric_value=0.52),
            _mkiter(2, delta=0.10, metric_value=0.60),
            _mkiter(3, delta=0.05, metric_value=0.55),
        ]
        summary = compile_hypothesis("h1", iterations, "maximize")

        assert summary.baseline_comparison is not None
        assert summary.baseline_comparison.best_improvement == 0.10

    def test_best_improvement_minimize_direction(self) -> None:
        """Minimize: best_improvement = min(delta)."""
        iterations = [
            _mkiter(1, delta=-0.02, metric_value=0.48),
            _mkiter(2, delta=-0.10, metric_value=0.40),
            _mkiter(3, delta=-0.05, metric_value=0.45),
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.baseline_comparison is not None
        assert summary.baseline_comparison.best_improvement == -0.10

    def test_overall_trend_improved(self) -> None:
        """Only favourable deltas → 'improved'."""
        iterations = [
            _mkiter(1, delta=-0.05, metric_value=0.45),
            _mkiter(2, delta=-0.08, metric_value=0.42),
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.baseline_comparison is not None
        assert summary.baseline_comparison.overall_trend == "improved"

    def test_overall_trend_worsened(self) -> None:
        """Only unfavourable deltas → 'worsened'."""
        iterations = [
            _mkiter(1, delta=0.05, metric_value=0.55),
            _mkiter(2, delta=0.02, metric_value=0.52),
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.baseline_comparison is not None
        assert summary.baseline_comparison.overall_trend == "worsened"

    def test_overall_trend_mixed(self) -> None:
        """Both favourable and unfavourable → 'mixed'."""
        iterations = [
            _mkiter(1, delta=-0.05, metric_value=0.45),  # favourable (minimize)
            _mkiter(2, delta=0.03, metric_value=0.53),  # unfavourable
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")

        assert summary.baseline_comparison is not None
        assert summary.baseline_comparison.overall_trend == "mixed"

    def test_best_metric_value_maximize(self) -> None:
        """Maximize: best_metric_value is the max among successful."""
        iterations = [
            _mkiter(1, delta=0.02, metric_value=0.52),
            _mkiter(2, delta=0.10, metric_value=0.60),
            _mkiter(3, delta=0.05, metric_value=0.55),
        ]
        summary = compile_hypothesis("h1", iterations, "maximize")
        assert summary.best_metric_value == 0.60

    def test_best_metric_value_minimize(self) -> None:
        """Minimize: best_metric_value is the min among successful."""
        iterations = [
            _mkiter(1, delta=-0.02, metric_value=0.48),
            _mkiter(2, delta=-0.10, metric_value=0.40),
            _mkiter(3, delta=-0.05, metric_value=0.45),
        ]
        summary = compile_hypothesis("h1", iterations, "minimize")
        assert summary.best_metric_value == 0.40

    def test_all_crashed_baseline_comparison_none(self) -> None:
        """All crashed → BaselineComparison is None."""
        iterations = [
            _mkiter(1, status="failed"),
            _mkiter(2, status="timeout"),
        ]
        summary = compile_hypothesis("h1", iterations, "maximize")

        assert summary.baseline_comparison is None
        assert summary.best_metric_value is None


# ---------------------------------------------------------------------------
# RC-11: compile_all tests
# ---------------------------------------------------------------------------


class TestCompileAll:
    """compile_all discovery and integration."""

    def test_multiple_hypotheses_compiled(self) -> None:
        """Multiple hypothesis directories are all compiled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Hypothesis 1
            h1_dir = exp_dir / "h1"
            h1_dir.mkdir()
            (h1_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\tbaseline_value\t"
                "delta\tstatus\ttimestamp\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            _write_program_md(h1_dir, "minimize")

            # Hypothesis 2
            h2_dir = exp_dir / "h2"
            h2_dir.mkdir()
            (h2_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\tbaseline_value\t"
                "delta\tstatus\ttimestamp\n"
                "1\taccuracy\t0.92\t0.90\t0.02\tsuccess\t2026-01-01T00:00:00Z\n"
                "2\taccuracy\t0.95\t0.92\t0.03\tsuccess\t2026-01-01T00:05:00Z\n",
                encoding="utf-8",
            )
            _write_program_md(h2_dir, "maximize")

            reader = TsvResultsReader(exp_dir)
            report = compile_all(reader, exp_dir)

            assert len(report.hypotheses) == 2
            assert report.generated_at  # non-empty

            h1_summary = next(h for h in report.hypotheses if h.hypothesis_id == "h1")
            h2_summary = next(h for h in report.hypotheses if h.hypothesis_id == "h2")

            assert h1_summary.status == "improved"  # minimize, delta -0.05
            assert h2_summary.status == "improved"  # maximize, both positive

    def test_missing_tsv_skipped(self) -> None:
        """Hypothesis dir without results.tsv is silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Has TSV
            h1_dir = exp_dir / "h1"
            h1_dir.mkdir()
            (h1_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\tbaseline_value\t"
                "delta\tstatus\ttimestamp\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            _write_program_md(h1_dir)

            # No TSV at all
            h2_dir = exp_dir / "h2"
            h2_dir.mkdir()

            reader = TsvResultsReader(exp_dir)
            report = compile_all(reader, exp_dir)

            assert len(report.hypotheses) == 1
            assert report.hypotheses[0].hypothesis_id == "h1"

    def test_empty_tsv_graceful(self) -> None:
        """Hypothesis with empty TSV produces 'no_iterations' status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            h1_dir = exp_dir / "h1"
            h1_dir.mkdir()
            (h1_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\tbaseline_value\t"
                "delta\tstatus\ttimestamp\n",
                encoding="utf-8",
            )
            _write_program_md(h1_dir)

            reader = TsvResultsReader(exp_dir)
            report = compile_all(reader, exp_dir)

            assert len(report.hypotheses) == 1
            assert report.hypotheses[0].status == "no_iterations"

    def test_no_experiments_directory_returns_empty(self) -> None:
        """Missing experiments directory → empty CompiledReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"
            reader = TsvResultsReader(nonexistent)
            report = compile_all(reader, nonexistent)

            assert report.hypotheses == []
            assert report.generated_at

    def test_direction_falls_back_to_maximize(self) -> None:
        """Missing program.md → direction defaults to 'maximize'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            h1_dir = exp_dir / "h1"
            h1_dir.mkdir()
            # Positive delta — should be kept under maximize (default)
            (h1_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\tbaseline_value\t"
                "delta\tstatus\ttimestamp\n"
                "1\tval_loss\t0.60\t0.50\t0.10\tsuccess\t2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            # No program.md

            reader = TsvResultsReader(exp_dir)
            report = compile_all(reader, exp_dir)

            assert len(report.hypotheses) == 1
            assert report.hypotheses[0].status == "improved"

    def test_non_directory_entries_skipped(self) -> None:
        """Files (not directories) in experiments dir are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            h1_dir = exp_dir / "h1"
            h1_dir.mkdir()
            (h1_dir / "results.tsv").write_text(
                "iteration\tmetric_name\tmetric_value\tbaseline_value\t"
                "delta\tstatus\ttimestamp\n"
                "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-01-01T00:00:00Z\n",
                encoding="utf-8",
            )
            _write_program_md(h1_dir, "minimize")

            # A plain file (not a directory) — should be skipped
            (exp_dir / "README.md").write_text("just a file", encoding="utf-8")

            reader = TsvResultsReader(exp_dir)
            report = compile_all(reader, exp_dir)

            assert len(report.hypotheses) == 1
