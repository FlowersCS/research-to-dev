"""Tests for ResultsWriter — TSV creation, header, row append, and schema
validation per D10 spec.

Covers AE-06: header creation on first write, row append on subsequent
writes, schema column order, and invalid status rejection.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from research_to_dev.experiment.results import ResultsWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_tsv(path: Path) -> list[list[str]]:
    """Read a TSV file and return rows as lists of strings (no stripping)."""
    content = path.read_text(encoding="utf-8").rstrip("\n")
    if not content:
        return []
    return [line.split("\t") for line in content.split("\n")]


# ---------------------------------------------------------------------------
# AE-08: ResultsWriter tests
# ---------------------------------------------------------------------------


class TestResultsWriterHeader:
    """RL-01, RL-02: Header creation on first write."""

    def test_creates_file_with_header_on_first_append(self) -> None:
        """First append creates the file with the correct header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)

            assert not tsv_path.exists()
            writer.append(1, "val_loss", 0.45, 0.50, -0.05, "success")
            assert tsv_path.exists()

            rows = _read_tsv(tsv_path)
            assert len(rows) == 2  # header + 1 data row
            assert rows[0] == [
                "iteration",
                "metric_name",
                "metric_value",
                "baseline_value",
                "delta",
                "status",
                "timestamp",
            ]

    def test_header_is_written_exactly_once(self) -> None:
        """Multiple appends do not duplicate the header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)

            writer.append(1, "val_loss", 0.45, 0.50, -0.05, "success")
            writer.append(2, "val_loss", 0.42, 0.45, -0.03, "success")

            rows = _read_tsv(tsv_path)
            assert len(rows) == 3  # header + 2 data rows
            assert rows[0] == [
                "iteration",
                "metric_name",
                "metric_value",
                "baseline_value",
                "delta",
                "status",
                "timestamp",
            ]

    def test_creates_parent_directories(self) -> None:
        """Missing parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "subdir" / "nested" / "results.tsv"
            writer = ResultsWriter(tsv_path)

            assert not tsv_path.parent.exists()
            writer.append(1, "accuracy", 0.92, 0.90, 0.02, "success")
            assert tsv_path.parent.exists()
            assert tsv_path.exists()


class TestResultsWriterRows:
    """RL-01, RL-03: Data row format and appending."""

    def test_data_row_has_correct_column_count(self) -> None:
        """Data row has exactly 7 columns matching D10."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)

            writer.append(1, "val_loss", 0.45, 0.50, -0.05, "success")

            rows = _read_tsv(tsv_path)
            assert len(rows) == 2  # header + data
            data = rows[1]
            assert len(data) == 7

    def test_data_row_contains_correct_values(self) -> None:
        """Data row values match what was passed in."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)

            writer.append(3, "accuracy", 0.92, 0.88, 0.04, "success")

            rows = _read_tsv(tsv_path)
            data = rows[1]
            assert data[0] == "3"  # iteration
            assert data[1] == "accuracy"  # metric_name
            assert data[2] == "0.92"  # metric_value
            assert data[3] == "0.88"  # baseline_value
            assert data[4] == "0.04"  # delta
            assert data[5] == "success"  # status
            # Timestamp exists and is non-empty
            assert data[6] != ""
            # Basic ISO format check: contains 'T' separator
            assert "T" in data[6]
            # Has timezone info (UTC: +00:00 or Z)
            assert "+00:00" in data[6] or data[6].endswith("Z")

    def test_appends_without_overwriting_existing_rows(self) -> None:
        """Existing rows are preserved when appending new rows (RL-03)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)

            writer.append(1, "val_loss", 0.50, 0.55, -0.05, "success")
            writer.append(2, "val_loss", 0.48, 0.50, -0.02, "success")

            rows = _read_tsv(tsv_path)
            assert len(rows) == 3  # header + 2 data rows
            # First data row unchanged
            assert rows[1][0] == "1"
            assert rows[1][1] == "val_loss"
            assert rows[1][2] == "0.5"
            assert rows[1][3] == "0.55"
            assert rows[1][4] == "-0.05"
            assert rows[1][5] == "success"
            # Second data row
            assert rows[2][0] == "2"
            assert rows[2][1] == "val_loss"
            assert rows[2][2] == "0.48"
            assert rows[2][3] == "0.5"
            assert rows[2][4] == "-0.02"
            assert rows[2][5] == "success"

    def test_positive_delta_formatting(self) -> None:
        """Positive delta is written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)

            writer.append(1, "accuracy", 0.95, 0.90, 0.05, "success")
            rows = _read_tsv(tsv_path)
            assert rows[1][4] == "0.05"

    def test_negative_delta_formatting(self) -> None:
        """Negative delta is written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)

            writer.append(1, "val_loss", 0.45, 0.50, -0.05, "failed")
            rows = _read_tsv(tsv_path)
            assert rows[1][4] == "-0.05"


class TestResultsWriterStatusValidation:
    """RL-04: Status must be one of success, failed, timeout."""

    def test_success_status_accepted(self) -> None:
        """'success' is a valid status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)
            writer.append(1, "val_loss", 0.45, 0.50, -0.05, "success")
            rows = _read_tsv(tsv_path)
            assert rows[1][5] == "success"

    def test_failed_status_accepted(self) -> None:
        """'failed' is a valid status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)
            writer.append(1, "val_loss", 0.45, 0.50, -0.05, "failed")
            rows = _read_tsv(tsv_path)
            assert rows[1][5] == "failed"

    def test_timeout_status_accepted(self) -> None:
        """'timeout' is a valid status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)
            writer.append(1, "val_loss", 0.45, 0.50, -0.05, "timeout")
            rows = _read_tsv(tsv_path)
            assert rows[1][5] == "timeout"

    def test_invalid_status_raises_value_error(self) -> None:
        """Invalid status raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)
            with pytest.raises(ValueError, match="Invalid status"):
                writer.append(1, "val_loss", 0.45, 0.50, -0.05, "excellent")

    def test_empty_status_raises_value_error(self) -> None:
        """Empty status raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)
            with pytest.raises(ValueError, match="Invalid status"):
                writer.append(1, "val_loss", 0.45, 0.50, -0.05, "")

    def test_all_valid_statuses_accepted(self) -> None:
        """All three valid statuses work without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"
            writer = ResultsWriter(tsv_path)

            writer.append(1, "loss", 1.0, 1.0, 0.0, "success")
            writer.append(2, "loss", 1.0, 1.0, 0.0, "failed")
            writer.append(3, "loss", 1.0, 1.0, 0.0, "timeout")

            rows = _read_tsv(tsv_path)
            assert len(rows) == 4  # header + 3 data rows
            assert rows[1][5] == "success"
            assert rows[2][5] == "failed"
            assert rows[3][5] == "timeout"


class TestResultsWriterIdempotency:
    """RL-02, RL-03: File creation and append behavior."""

    def test_multiple_writers_to_same_file_share_header(self) -> None:
        """Second writer reuses existing file, does not write header again."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "results.tsv"

            writer1 = ResultsWriter(tsv_path)
            writer1.append(1, "val_loss", 0.50, 0.55, -0.05, "success")

            writer2 = ResultsWriter(tsv_path)
            writer2.append(2, "val_loss", 0.48, 0.50, -0.02, "success")

            rows = _read_tsv(tsv_path)
            assert len(rows) == 3  # header + 2 data rows
            # Verify header only appears once
            header_lines = [r for r in rows if r[0] == "iteration"]
            assert len(header_lines) == 1
