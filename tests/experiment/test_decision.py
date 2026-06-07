"""Tests for decide_keep() — keep/discard decision logic with direction-aware
comparison (AE-17)."""

from __future__ import annotations

import pytest

from research_to_dev.experiment.metrics import decide_keep


class TestDecideKeepMaximize:
    """KD-02: For direction="maximize", keep when metric >= baseline."""

    def test_improvement_is_keep(self) -> None:
        """Higher value → keep (True)."""
        assert decide_keep(0.9, 0.8, "maximize") is True

    def test_worsening_is_discard(self) -> None:
        """Lower value → discard (False)."""
        assert decide_keep(0.7, 0.8, "maximize") is False

    def test_equal_values_are_keep(self) -> None:
        """Equal value → keep (True) — do not discard progress it cannot yet beat."""
        assert decide_keep(0.8, 0.8, "maximize") is True

    def test_large_difference_improvement(self) -> None:
        """Large positive delta is still keep."""
        assert decide_keep(0.99, 0.50, "maximize") is True

    def test_tiny_difference_worsening(self) -> None:
        """Even a tiny negative delta is discard."""
        assert decide_keep(0.7999, 0.8, "maximize") is False


class TestDecideKeepMinimize:
    """KD-03: For direction="minimize", keep when metric <= baseline."""

    def test_improvement_is_keep(self) -> None:
        """Lower value → keep (True)."""
        assert decide_keep(0.3, 0.5, "minimize") is True

    def test_worsening_is_discard(self) -> None:
        """Higher value → discard (False)."""
        assert decide_keep(0.6, 0.5, "minimize") is False

    def test_equal_values_are_keep(self) -> None:
        """Equal value → keep (True)."""
        assert decide_keep(0.5, 0.5, "minimize") is True

    def test_large_difference_improvement(self) -> None:
        """Large negative delta is still keep."""
        assert decide_keep(0.01, 0.50, "minimize") is True

    def test_tiny_difference_worsening(self) -> None:
        """Even a tiny positive delta is discard."""
        assert decide_keep(0.5001, 0.5, "minimize") is False


class TestDecideKeepEdgeCases:
    """Boundary and error cases for decide_keep."""

    def test_zero_baseline_maximize(self) -> None:
        """Baseline can be zero for maximize direction."""
        assert decide_keep(0.0, 0.0, "maximize") is True
        assert decide_keep(0.1, 0.0, "maximize") is True

    def test_zero_baseline_minimize(self) -> None:
        """Baseline can be zero for minimize direction."""
        assert decide_keep(0.0, 0.0, "minimize") is True
        assert decide_keep(0.1, 0.0, "minimize") is False

    def test_invalid_direction_raises_value_error(self) -> None:
        """Unknown direction string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid direction"):
            decide_keep(0.5, 0.5, "keep")

    def test_invalid_direction_empty_string(self) -> None:
        """Empty direction raises ValueError."""
        with pytest.raises(ValueError):
            decide_keep(0.5, 0.5, "")

    def test_baseline_update_scenario(self) -> None:
        """D9: baseline updates after keep — this function only compares;
        the runner handles the baseline update.
        
        After a keep, the runner would update baseline to current_value,
        then the next comparison uses the updated baseline."""
        # Iteration 1: improve from 0.5 → 0.4 (minimize), keep
        assert decide_keep(0.4, 0.5, "minimize") is True
        # Runner updates baseline to 0.4
        # Iteration 2: try to beat 0.4 — 0.45 is worse, discard
        assert decide_keep(0.45, 0.4, "minimize") is False
        # Iteration 3: improve to 0.38 — keep
        assert decide_keep(0.38, 0.4, "minimize") is True
