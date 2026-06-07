"""Tests for MetricRegistry — built-in defaults, config.yaml override, and
fail-loud regex validation at load time (ES-01A through ES-01D)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from research_to_dev.experiment.metrics import MetricRegistry


class TestBuiltInMetrics:
    """ES-01A: Built-in metric resolution."""

    def test_built_in_val_loss_resolved(self) -> None:
        """val_loss is available as a built-in metric."""
        registry = MetricRegistry()
        pattern = registry.get("val_loss")
        assert pattern is not None
        assert "val_loss" in pattern

    def test_built_in_accuracy_resolved(self) -> None:
        """accuracy is available as a built-in metric."""
        registry = MetricRegistry()
        pattern = registry.get("accuracy")
        assert pattern is not None
        assert "accuracy" in pattern

    def test_unknown_metric_returns_none(self) -> None:
        """Unknown metric returns None."""
        registry = MetricRegistry()
        assert registry.get("not_a_metric") is None

    def test_validate_metric_true_for_builtin(self) -> None:
        """validate_metric returns True for known built-ins."""
        registry = MetricRegistry()
        assert registry.validate_metric("val_loss") is True
        assert registry.validate_metric("accuracy") is True

    def test_validate_metric_false_for_unknown(self) -> None:
        """validate_metric returns False for unknown metric."""
        registry = MetricRegistry()
        assert registry.validate_metric("custom_auc") is False

    def test_list_all_includes_six_builtins(self) -> None:
        """list_all returns all 6 built-in metrics sorted."""
        registry = MetricRegistry()
        names = registry.list_all()
        assert len(names) >= 6
        assert "accuracy" in names
        assert "val_loss" in names
        assert names == sorted(names)


class TestConfigOverride:
    """ES-01B: Config override takes precedence."""

    def test_config_override_takes_precedence(self) -> None:
        """Config.yaml pattern overrides built-in for matching metric name."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("metrics:\n  val_loss: 'custom_val_loss_pattern'\n")
            config_path = f.name

        try:
            registry = MetricRegistry(config_path=config_path)
            pattern = registry.get("val_loss")
            assert pattern == "custom_val_loss_pattern"
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_custom_metric_not_in_builtin(self) -> None:
        """ES-01D: Custom metric from config.yaml is available."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(
                "metrics:\n"
                "  custom_auc: 'auc[:_\\s=]*([\\d.]+)'\n"
            )
            config_path = f.name

        try:
            registry = MetricRegistry(config_path=config_path)
            assert registry.validate_metric("custom_auc") is True
            pattern = registry.get("custom_auc")
            assert "auc" in pattern
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_config_missing_file_does_not_crash(self) -> None:
        """Missing config.yaml is silently ignored (config is optional)."""
        registry = MetricRegistry(config_path="/nonexistent/config.yaml")
        # Should still have built-ins
        assert registry.get("val_loss") is not None

    def test_empty_config_does_not_crash(self) -> None:
        """Empty config.yaml does not break anything."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            config_path = f.name

        try:
            registry = MetricRegistry(config_path=config_path)
            assert registry.get("val_loss") is not None
        finally:
            Path(config_path).unlink(missing_ok=True)


class TestInvalidRegexFailLoud:
    """ES-01C: Invalid regex fails loud on config load (D12)."""

    def test_invalid_regex_raises_value_error(self) -> None:
        """Invalid regex pattern raises ValueError with actionable message."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("metrics:\n  bad_metric: '[invalid('\n")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid regex"):
                MetricRegistry(config_path=config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_invalid_regex_message_includes_metric_name(self) -> None:
        """Error message includes the metric name for debugging."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("metrics:\n  bad_metric: '[invalid('\n")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="bad_metric"):
                MetricRegistry(config_path=config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_none_config_path_uses_builtins_only(self) -> None:
        """When config_path is None, only built-ins are used."""
        registry = MetricRegistry(config_path=None)
        assert len(registry.list_all()) == 6  # only built-ins


class TestMetricRegistryExport:
    """to_configs() exports MetricConfig list."""

    def test_to_configs_returns_metricconfigs(self) -> None:
        """to_configs returns list of MetricConfig objects."""
        from research_to_dev.shared.config import MetricConfig

        registry = MetricRegistry()
        configs = registry.to_configs()
        assert len(configs) == 6
        assert all(isinstance(c, MetricConfig) for c in configs)
        names = [c.name for c in configs]
        assert "val_loss" in names
        assert "accuracy" in names

    def test_to_configs_with_custom_metrics(self) -> None:
        """to_configs includes custom overrides."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("metrics:\n  custom_auc: 'auc[:_\\s=]*([\\d.]+)'\n")
            config_path = f.name

        try:
            registry = MetricRegistry(config_path=config_path)
            configs = registry.to_configs()
            names = [c.name for c in configs]
            assert "custom_auc" in names
            assert "val_loss" in names
            # custom_auc replaces nothing, just adds
        finally:
            Path(config_path).unlink(missing_ok=True)


class TestExtract:
    """AE-16: MetricRegistry.extract(text, metric_name) → float | None."""

    def test_extract_returns_float_for_match(self) -> None:
        """extract returns the float value when the pattern matches."""
        registry = MetricRegistry()
        result = registry.extract("accuracy: 0.92", "accuracy")
        assert result == 0.92

    def test_extract_val_loss_near_prefix(self) -> None:
        """extract works when the metric name is part of a longer line."""
        registry = MetricRegistry()
        result = registry.extract("val_loss = 0.345", "val_loss")
        assert result == 0.345

    def test_extract_f1_with_surrounding_text(self) -> None:
        """extract finds the metric value embedded in surrounding output."""
        registry = MetricRegistry()
        text = "Epoch 10: loss=0.123, f1: 0.87, bleu=32.1"
        result = registry.extract(text, "f1")
        assert result == 0.87

    def test_extract_no_match_returns_none(self) -> None:
        """extract returns None when the pattern is not found in the text."""
        registry = MetricRegistry()
        result = registry.extract("hello world", "accuracy")
        assert result is None

    def test_extract_partial_name_not_matched(self) -> None:
        """extract does not match a partial metric name."""
        registry = MetricRegistry()
        result = registry.extract("my_accuracy: 0.99", "accuracy")
        # The pattern "accuracy[:\s=]*([\d.]+)" will match "accuracy: 0.99"
        # inside "my_accuracy: 0.99" — the substring will be matched.
        assert result == 0.99

    def test_extract_multiple_matches_returns_first(self) -> None:
        """extract returns the first match when the metric appears more
        than once in the text."""
        registry = MetricRegistry()
        text = "accuracy: 0.90\n...more output...\naccuracy: 0.92"
        result = registry.extract(text, "accuracy")
        assert result == 0.90

    def test_extract_unknown_metric_raises_value_error(self) -> None:
        """extract raises ValueError when the metric_name is not registered."""
        registry = MetricRegistry()
        with pytest.raises(ValueError, match="Unknown metric"):
            registry.extract("some text", "not_a_metric")

    def test_extract_from_stdout_like_string(self) -> None:
        """extract handles multi-line stdout-like strings."""
        registry = MetricRegistry()
        stdout = (
            "Training complete!\n"
            "accuracy: 0.95\n"
            "loss: 0.05\n"
        )
        assert registry.extract(stdout, "accuracy") == 0.95
        assert registry.extract(stdout, "loss") == 0.05

    def test_extract_perplexity_integer_like(self) -> None:
        """extract handles perplexity values that look like integers."""
        registry = MetricRegistry()
        result = registry.extract("perplexity: 42", "perplexity")
        assert result == 42.0

    def test_extract_bleu_with_dot_path(self) -> None:
        """extract handles BLEU scores like 32.1."""
        registry = MetricRegistry()
        result = registry.extract("bleu = 32.1", "bleu")
        assert result == 32.1
