"""Tests for TsvResultsReader — TSV parsing, header validation, malformed
row handling, and error edge cases per RC-07."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from research_to_dev.report.compilation import TsvResultsReader, IterationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_HEADER = (
    "iteration\tmetric_name\tmetric_value\t"
    "baseline_value\tdelta\tstatus\ttimestamp"
)


def _write_tsv(tsv_dir: Path, content: str, hypothesis_id: str = "abc123") -> Path:
    """Write a results.tsv file inside a hypothesis subdirectory."""
    hyp_dir = tsv_dir / hypothesis_id
    hyp_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = hyp_dir / "results.tsv"
    tsv_path.write_text(content, encoding="utf-8")
    return tsv_path


# ---------------------------------------------------------------------------
# RC-07: TsvResultsReader
# ---------------------------------------------------------------------------


class TestTsvResultsReaderValid:
    """Parsing valid TSV files."""

    def test_parses_valid_tsv_with_multiple_rows(self) -> None:
        """Valid 7-column TSV with multiple data rows returns correct
        IterationResult objects."""
        content = (
            f"{_VALID_HEADER}\n"
            "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"
            "2\tval_loss\t0.42\t0.45\t-0.03\tsuccess\t2026-06-07T14:05:00Z\n"
            "3\tval_loss\t0.48\t0.42\t0.06\tfailed\t2026-06-07T14:10:00Z\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            assert len(results) == 3

            assert results[0] == IterationResult(
                iteration=1,
                metric_name="val_loss",
                metric_value=0.45,
                baseline_value=0.50,
                delta=-0.05,
                status="success",
                timestamp="2026-06-07T14:00:00Z",
            )

            assert results[1].iteration == 2
            assert results[1].delta == -0.03
            assert results[2].status == "failed"

    def test_single_row_tsv(self) -> None:
        """Single data row is parsed correctly."""
        content = (
            f"{_VALID_HEADER}\n"
            "1\taccuracy\t0.92\t0.90\t0.02\tsuccess\t2026-06-07T14:00:00Z\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            assert len(results) == 1
            assert results[0].metric_name == "accuracy"
            assert results[0].metric_value == 0.92


class TestTsvResultsReaderErrors:
    """Error and edge case handling."""

    def test_missing_results_tsv_raises_filenotfounderror(self) -> None:
        """Missing results.tsv raises FileNotFoundError with clear message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            # Create the hypothesis dir but NOT results.tsv
            (exp_dir / "abc123").mkdir(parents=True)

            reader = TsvResultsReader(exp_dir)
            with pytest.raises(FileNotFoundError, match="results.tsv not found"):
                reader.read("abc123")

    def test_nonexistent_hypothesis_dir_raises(self) -> None:
        """Hypothesis directory that doesn't exist raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            reader = TsvResultsReader(exp_dir)
            with pytest.raises(FileNotFoundError, match="results.tsv not found"):
                reader.read("nonexistent")

    def test_empty_tsv_header_only_returns_empty_list(self) -> None:
        """TSV with header but no data rows returns empty list + warning."""
        content = _VALID_HEADER + "\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            assert results == []
            assert len(results) == 0

    def test_completely_empty_file_returns_empty_list(self) -> None:
        """Empty TSV file (no content) returns empty list."""
        content = ""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            assert results == []

    def test_malformed_row_skipped_with_warning(self) -> None:
        """Rows with wrong column count are skipped, valid rows parsed."""
        content = (
            f"{_VALID_HEADER}\n"
            "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"
            "2\tbad_row\t0.42\t0.45\n"  # only 4 columns
            "3\tval_loss\t0.48\t0.42\t0.06\tfailed\t2026-06-07T14:10:00Z\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            # Malformed row (line 2) should be skipped
            assert len(results) == 2
            assert results[0].iteration == 1
            assert results[1].iteration == 3

    def test_type_coercion_failure_skipped(self) -> None:
        """Rows where numeric fields can't be parsed are skipped."""
        content = (
            f"{_VALID_HEADER}\n"
            "1\tval_loss\tnot_a_number\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"
            "2\tval_loss\t0.42\t0.45\t-0.03\tsuccess\t2026-06-07T14:05:00Z\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            assert len(results) == 1
            assert results[0].iteration == 2

    def test_timeout_status_rows_parsed(self) -> None:
        """Rows with 'timeout' status are parsed as regular rows."""
        content = (
            f"{_VALID_HEADER}\n"
            "1\tval_loss\t0.45\t0.50\t-0.05\ttimeout\t2026-06-07T14:00:00Z\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            assert len(results) == 1
            assert results[0].status == "timeout"

    def test_negative_iteration_values_parsed(self) -> None:
        """Negative numeric values are correctly parsed."""
        content = (
            f"{_VALID_HEADER}\n"
            "1\tloss\t-0.5\t-0.3\t-0.2\tsuccess\t2026-06-07T14:00:00Z\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            assert len(results) == 1
            assert results[0].metric_value == -0.5
            assert results[0].delta == -0.2

    def test_empty_lines_in_tsv_skipped(self) -> None:
        """Blank lines between data rows are skipped."""
        content = (
            f"{_VALID_HEADER}\n"
            "1\tval_loss\t0.45\t0.50\t-0.05\tsuccess\t2026-06-07T14:00:00Z\n"
            "\n"
            "2\tval_loss\t0.42\t0.45\t-0.03\tsuccess\t2026-06-07T14:05:00Z\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)
            _write_tsv(exp_dir, content)

            reader = TsvResultsReader(exp_dir)
            results = reader.read("abc123")

            assert len(results) == 2
