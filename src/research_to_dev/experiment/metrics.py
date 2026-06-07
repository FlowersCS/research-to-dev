"""Metric registry with built-in defaults and optional config.yaml override.

Follows D5 and D12: built-in metric patterns for common ML metrics (val_loss,
accuracy, loss, f1, bleu, perplexity).  Per-project overrides via
``.research-to-dev/config.yaml``.  Regex validation happens at load time —
invalid patterns fail loud immediately, never at runtime (D12).
"""

from __future__ import annotations

import re
from typing import ClassVar

from research_to_dev.experiment.config import load_config_yaml
from research_to_dev.shared.config import MetricConfig


class MetricRegistry:
    """Registry of metric patterns — built-in defaults merged with per-project config.yaml overrides.

    Built-in metrics are always available.  If ``config.yaml`` defines a
    metric with the same name, the config version silently replaces the
    built-in (D5).  Custom metrics only defined in config.yaml are also
    available (ES-01D).

    Usage::

        registry = MetricRegistry(config_path=".research-to-dev/config.yaml")
        pattern = registry.get("val_loss")  # str | None
        is_valid = registry.validate_metric("accuracy")  # bool
        all_names = registry.list_all()  # list[str]
    """

    BUILTINS: ClassVar[dict[str, str]] = {
        "val_loss": r"val_loss[:\s=]*([\d.]+)",
        "accuracy": r"accuracy[:\s=]*([\d.]+)",
        "loss": r"loss[:\s=]*([\d.]+)",
        "f1": r"f1[:\s=]*([\d.]+)",
        "bleu": r"bleu[:\s=]*([\d.]+)",
        "perplexity": r"perplexity[:\s=]*([\d.]+)",
    }

    def __init__(self, config_path: str | None = None) -> None:
        """Initialise the registry with built-in patterns and optional config.yaml overrides.

        Args:
            config_path: Optional path to ``.research-to-dev/config.yaml``.
                If provided and the file exists, custom metric definitions
                are merged on top of built-in defaults.
        """
        self._patterns: dict[str, str] = dict(self.BUILTINS)

        if config_path:
            config = load_config_yaml(config_path)
            custom_metrics = config.get("metrics", {})
            if isinstance(custom_metrics, dict):
                for name, pattern in custom_metrics.items():
                    try:
                        re.compile(str(pattern))
                    except re.error as exc:
                        msg = (
                            f"Invalid regex in config.yaml for metric "
                            f"'{name}': pattern '{pattern}' is not a "
                            f"valid regex ({exc}).\n"
                            f"Fix the pattern in config.yaml and try again."
                        )
                        raise ValueError(msg) from exc
                    self._patterns[name] = str(pattern)

    # -- public API --------------------------------------------------------

    def get(self, name: str) -> str | None:
        """Return the regex pattern for a metric, or ``None`` if not found.

        Args:
            name: Metric identifier (e.g. ``"val_loss"``).

        Returns:
            The regex pattern string, or ``None`` if the metric is unknown.
        """
        return self._patterns.get(name)

    def validate_metric(self, name: str) -> bool:
        """Check whether a metric name is registered.

        Args:
            name: Metric identifier to validate.

        Returns:
            ``True`` if the metric exists in the registry.
        """
        return name in self._patterns

    def list_all(self) -> list[str]:
        """Return all registered metric names (built-ins + custom overrides).

        Returns:
            Sorted list of metric name strings.
        """
        return sorted(self._patterns.keys())

    # -- helpers -----------------------------------------------------------

    def to_configs(self) -> list[MetricConfig]:
        """Export the registry's current state as ``MetricConfig`` objects."""
        return [
            MetricConfig(name=name, pattern=pat)
            for name, pat in sorted(self._patterns.items())
        ]
