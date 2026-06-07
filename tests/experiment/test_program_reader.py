"""Tests for program_reader.py — parsing program.md, duration conversion,
validation failures, and direction inference.

Covers AE-01 through AE-05: valid parse, missing fields (baseline,
run_command, coding_agent_model), invalid YAML, direction inference,
and duration conversions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from research_to_dev.experiment.program_reader import (
    ProgramSpec,
    parse_duration,
    parse_program_md,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_program_md(frontmatter: dict, body: str = "## Content\n\nHello world.\n") -> Path:
    """Write a temporary program.md with the given frontmatter and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    yaml_text = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip()
    content = f"---\n{yaml_text}\n---\n\n{body}"
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _valid_frontmatter() -> dict:
    """Return a valid frontmatter dict with all mandatory fields."""
    return {
        "hypothesis_id": "abc123def456",
        "target_metric": "val_loss",
        "success_criteria": "val_loss <= 0.45",
        "time_budget": "30m",
        "max_iterations": 10,
        "baseline": {"val_loss": 0.72},
        "run_command": "python train.py --epochs 10",
        "coding_agent_model": "opencode-go/deepseek-v4-pro",
    }


# ---------------------------------------------------------------------------
# AE-07: parse_program_md tests
# ---------------------------------------------------------------------------


class TestParseProgramMdValid:
    """PR-01, PR-02: Valid program.md parsed correctly."""

    def test_parses_all_required_fields(self) -> None:
        """All mandatory fields are extracted correctly."""
        fm = _valid_frontmatter()
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.hypothesis_id == "abc123def456"
            assert spec.target_metric == "val_loss"
            assert spec.success_criteria == "val_loss <= 0.45"
            assert spec.time_budget == "30m"
            assert spec.max_iterations == 10
            assert spec.baseline == {"val_loss": 0.72}
            assert spec.run_command == "python train.py --epochs 10"
            assert spec.coding_agent_model == "opencode-go/deepseek-v4-pro"
        finally:
            path.unlink(missing_ok=True)

    def test_parses_time_budget_seconds(self) -> None:
        """time_budget_seconds is correctly converted (D5)."""
        fm = _valid_frontmatter()
        fm["time_budget"] = "2h"
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.time_budget_seconds == 7200  # 2 * 3600
        finally:
            path.unlink(missing_ok=True)

    def test_parses_body_content(self) -> None:
        """Markdown body is preserved."""
        body = "## My Experiment\n\nSome explanation here.\n"
        fm = _valid_frontmatter()
        path = _write_program_md(fm, body=body)
        try:
            spec = parse_program_md(path)
            assert spec.body == body
        finally:
            path.unlink(missing_ok=True)

    def test_program_spec_is_dataclass(self) -> None:
        """Returned object is a ProgramSpec dataclass."""
        fm = _valid_frontmatter()
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert isinstance(spec, ProgramSpec)
        finally:
            path.unlink(missing_ok=True)

    def test_baseline_values_are_floats(self) -> None:
        """Baseline dict values are converted to float."""
        fm = _valid_frontmatter()
        fm["baseline"] = {"val_accuracy": "0.92"}
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.baseline == {"val_accuracy": 0.92}
            assert isinstance(spec.baseline["val_accuracy"], float)
        finally:
            path.unlink(missing_ok=True)


class TestMissingFieldsFailLoud:
    """PR-03, PR-04, D3, D4, D12: Mandatory fields raise ValueError."""

    def test_missing_baseline_raises(self) -> None:
        """Missing baseline raises ValueError (D4)."""
        fm = _valid_frontmatter()
        del fm["baseline"]
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="baseline"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_run_command_raises(self) -> None:
        """Missing run_command raises ValueError (D3)."""
        fm = _valid_frontmatter()
        del fm["run_command"]
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="run_command"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_coding_agent_model_raises(self) -> None:
        """Missing coding_agent_model raises ValueError (D12)."""
        fm = _valid_frontmatter()
        del fm["coding_agent_model"]
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="coding_agent_model"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_hypothesis_id_raises(self) -> None:
        """Missing hypothesis_id raises ValueError."""
        fm = _valid_frontmatter()
        del fm["hypothesis_id"]
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="hypothesis_id"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_target_metric_raises(self) -> None:
        """Missing target_metric raises ValueError."""
        fm = _valid_frontmatter()
        del fm["target_metric"]
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="target_metric"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_multiple_fields_reports_all(self) -> None:
        """Error message lists all missing fields."""
        fm: dict = {}
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError) as exc_info:
                parse_program_md(path)
            msg = str(exc_info.value)
            assert "baseline" in msg
            assert "run_command" in msg
            assert "coding_agent_model" in msg
        finally:
            path.unlink(missing_ok=True)


class TestInvalidYaml:
    """PR spec: Invalid YAML in frontmatter fails loud."""

    def test_malformed_yaml_raises_value_error(self) -> None:
        """Malformed frontmatter YAML raises ValueError."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tmp.write("---\n{{{ bad: yaml:::\n---\n\nBody\n")
        tmp.close()
        path = Path(tmp.name)
        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_frontmatter_not_a_dict_raises(self) -> None:
        """Frontmatter that parses to a list raises ValueError."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tmp.write("---\n- item1\n- item2\n---\n\nBody\n")
        tmp.close()
        path = Path(tmp.name)
        try:
            with pytest.raises(ValueError, match="must be a YAML mapping"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_frontmatter_delimiters_raises(self) -> None:
        """File without --- delimiters raises ValueError."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tmp.write("Just some markdown\nno frontmatter.\n")
        tmp.close()
        path = Path(tmp.name)
        try:
            with pytest.raises(ValueError, match="delimited by '---'"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)


class TestDirectionInference:
    """PR-06, D6: Direction inferred from success_criteria operator."""

    def test_infers_minimize_from_lt(self) -> None:
        """`<` infers minimize."""
        fm = _valid_frontmatter()
        fm["success_criteria"] = "val_loss < 0.5"
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.direction == "minimize"
        finally:
            path.unlink(missing_ok=True)

    def test_infers_minimize_from_lte(self) -> None:
        """`<=` infers minimize."""
        fm = _valid_frontmatter()
        fm["success_criteria"] = "val_loss <= 0.45"
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.direction == "minimize"
        finally:
            path.unlink(missing_ok=True)

    def test_infers_maximize_from_gt(self) -> None:
        """`>` infers maximize."""
        fm = _valid_frontmatter()
        fm["success_criteria"] = "accuracy > 0.85"
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.direction == "maximize"
        finally:
            path.unlink(missing_ok=True)

    def test_infers_maximize_from_gte(self) -> None:
        """`>=` infers maximize."""
        fm = _valid_frontmatter()
        fm["success_criteria"] = "accuracy >= 0.80"
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.direction == "maximize"
        finally:
            path.unlink(missing_ok=True)

    def test_explicit_direction_overrides_inference(self) -> None:
        """Explicit direction field overrides inferred direction (D6)."""
        fm = _valid_frontmatter()
        fm["success_criteria"] = "val_loss < 0.5"  # would infer minimize
        fm["direction"] = "maximize"  # override
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.direction == "maximize"
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_direction_value_raises(self) -> None:
        """Invalid explicit direction raises ValueError."""
        fm = _valid_frontmatter()
        fm["direction"] = "sideways"
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="must be 'maximize' or 'minimize'"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_no_operator_raises(self) -> None:
        """success_criteria without operator raises ValueError."""
        fm = _valid_frontmatter()
        fm["success_criteria"] = "the model should work well"
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="Cannot infer direction"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)


class TestMaxIterationsValidation:
    """max_iterations must be a positive integer."""

    def test_valid_max_iterations(self) -> None:
        """Positive integer is accepted."""
        fm = _valid_frontmatter()
        fm["max_iterations"] = 5
        path = _write_program_md(fm)
        try:
            spec = parse_program_md(path)
            assert spec.max_iterations == 5
        finally:
            path.unlink(missing_ok=True)

    def test_zero_max_iterations_raises(self) -> None:
        """Zero max_iterations raises ValueError."""
        fm = _valid_frontmatter()
        fm["max_iterations"] = 0
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="positive integer"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_negative_max_iterations_raises(self) -> None:
        """Negative max_iterations raises ValueError."""
        fm = _valid_frontmatter()
        fm["max_iterations"] = -3
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="positive integer"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_string_max_iterations_raises(self) -> None:
        """Non-integer max_iterations raises ValueError."""
        fm = _valid_frontmatter()
        fm["max_iterations"] = "ten"
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="positive integer"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)


class TestBaselineValidation:
    """baseline must be a non-empty dict."""

    def test_baseline_is_empty_dict_raises(self) -> None:
        """Empty dict baseline raises ValueError."""
        fm = _valid_frontmatter()
        fm["baseline"] = {}
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="non-empty dict"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)

    def test_baseline_is_string_raises(self) -> None:
        """Non-dict baseline raises ValueError."""
        fm = _valid_frontmatter()
        fm["baseline"] = "0.72"
        path = _write_program_md(fm)
        try:
            with pytest.raises(ValueError, match="non-empty dict"):
                parse_program_md(path)
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AE-07: parse_duration tests
# ---------------------------------------------------------------------------


class TestParseDuration:
    """PR-05, D5: Duration parsing to integer seconds."""

    def test_minutes_only(self) -> None:
        """'30m' → 1800."""
        assert parse_duration("30m") == 1800

    def test_hours_only(self) -> None:
        """'2h' → 7200."""
        assert parse_duration("2h") == 7200

    def test_hours_and_minutes(self) -> None:
        """'1h30m' → 5400."""
        assert parse_duration("1h30m") == 5400

    def test_multi_digit_hours_and_minutes(self) -> None:
        """'24h59m' → 89940."""
        assert parse_duration("24h59m") == 24 * 3600 + 59 * 60

    def test_with_whitespace(self) -> None:
        """Whitespace is stripped."""
        assert parse_duration("  30m  ") == 1800

    def test_only_minutes_large(self) -> None:
        """'120m' → 7200."""
        assert parse_duration("120m") == 7200


class TestParseDurationInvalid:
    """Invalid duration formats raise ValueError."""

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_duration("")

    def test_whitespace_only_raises(self) -> None:
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_duration("   ")

    def test_missing_minutes_unit_raises(self) -> None:
        """'2h30' (missing 'm') raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("2h30")

    def test_seconds_format_raises(self) -> None:
        """'3600s' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("3600s")

    def test_garbage_input_raises(self) -> None:
        """Random text raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("not a duration")

    def test_negative_minutes_raises(self) -> None:
        """'-30m' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("-30m")

    def test_zero_seconds_raises(self) -> None:
        """'0m' raises ValueError (zero duration)."""
        with pytest.raises(ValueError, match="resolves to 0 seconds"):
            parse_duration("0m")

    def test_zero_hours_and_minutes_raises(self) -> None:
        """'0h0m' raises ValueError (zero duration)."""
        with pytest.raises(ValueError, match="resolves to 0 seconds"):
            parse_duration("0h0m")

    def test_reversed_order_raises(self) -> None:
        """'30m2h' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("30m2h")
