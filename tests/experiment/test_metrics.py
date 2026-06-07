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
