"""Results writer — appends iteration results to a TSV file per D10 schema.

Writes tab-separated rows with header::

    iteration  metric_name  metric_value  baseline_value  delta  status  timestamp

Creates the file (with header) on first write; appends on subsequent
writes (RL-01 through RL-04).

Usage::

    writer = ResultsWriter(Path("results.tsv"))
    writer.append(1, "val_loss", 0.45, 0.50, -0.05, "success")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

# Column order per D10 — must match exactly
_COLUMNS = [
    "iteration",
    "metric_name",
    "metric_value",
    "baseline_value",
    "delta",
    "status",
    "timestamp",
]

_VALID_STATUSES = frozenset({"success", "failed", "timeout"})


class ResultsWriter:
    """Appends iteration results to a TSV file.

    Constructor-injectable so tests can supply arbitrary paths without
    touching the real filesystem.

    Usage::

        writer = ResultsWriter(Path("experiments/abc123/results.tsv"))
        writer.append(1, "val_loss", 0.92, 0.90, 0.02, "success")
    """

    def __init__(self, tsv_path: Path) -> None:
        """Initialise with the target TSV path.

        The file is NOT created until the first call to ``append()``.
        """
        self._path = tsv_path

    def append(
        self,
        iteration: int,
        metric_name: str,
        value: float,
        baseline: float,
        delta: float,
        status: str,
    ) -> None:
        """Append an iteration result row to the TSV file.

        On first call, the file is created with a header row.  Subsequent
        calls append data rows.

        Args:
            iteration: 1-based iteration number.
            metric_name: Name of the metric evaluated.
            value: Observed metric value for this iteration.
            baseline: Baseline metric value for comparison.
            delta: Difference (value - baseline).
            status: One of ``"success"``, ``"failed"``, ``"timeout"``.

        Raises:
            ValueError: If ``status`` is not one of the valid values.
        """
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Must be one of: {', '.join(sorted(_VALID_STATUSES))}."
            )

        timestamp = datetime.now(timezone.utc).isoformat()

        # Write header if file doesn't exist yet (RL-02)
        if not self._path.exists() or self._path.stat().st_size == 0:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            header = "\t".join(_COLUMNS) + "\n"
            self._path.write_text(header, encoding="utf-8")

        # Append data row (RL-03)
        row = "\t".join(
            [
                str(iteration),
                metric_name,
                str(value),
                str(baseline),
                str(delta),
                status,
                timestamp,
            ]
        ) + "\n"

        with self._path.open("a", encoding="utf-8") as f:
            f.write(row)
