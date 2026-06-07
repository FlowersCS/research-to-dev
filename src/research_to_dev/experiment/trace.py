"""Trace reading — load ``PipelineTrace`` from JSON and extract hypotheses.

Used by the experiment setup command to locate a hypothesis by its
SHA-256 ID from the pipeline output.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_to_dev.cli.orchestrator import PipelineTrace
from research_to_dev.hypothesis.types import Hypothesis


def read_trace(path: str) -> PipelineTrace:
    """Load a ``PipelineTrace`` from a JSON file.

    Args:
        path: Filesystem path to the trace file (e.g.
            ``.research-to-dev/pipeline/trace.json``).

    Returns:
        A reconstructed ``PipelineTrace`` instance.

    Raises:
        FileNotFoundError: If the file does not exist (caller should catch
            and display an actionable message per Gap 2).
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Trace not found at {path}. "
            f"Run `research-to-dev analyze` first."
        )

    data = json.loads(file_path.read_text(encoding="utf-8"))
    return _reconstruct_trace(data)


def extract_hypotheses(trace: PipelineTrace) -> list[Hypothesis]:
    """Extract the ``Hypothesis`` list from a ``PipelineTrace``.

    Reads from ``trace.hypothesis.hypotheses`` if available (the
    HypothesisResult from the pipeline).  Returns the raw Hypothesis
    dataclass instances for searching by ID.

    Args:
        trace: A loaded ``PipelineTrace``.

    Returns:
        List of ``Hypothesis`` objects.  Empty list if no hypotheses exist.
    """
    hypos_raw = trace.hypotheses  # list[dict] from PipelineTrace
    if not hypos_raw:
        return []

    result: list[Hypothesis] = []
    for hd in hypos_raw:
        if not isinstance(hd, dict):
            continue
        result.append(
            Hypothesis(
                id=hd.get("id", ""),
                title=hd.get("title", ""),
                description=hd.get("description", ""),
                approach=hd.get("approach", ""),
                supporting_papers=hd.get("supporting_papers", []),
                target_metric=hd.get("target_metric", ""),
                expected_improvement=hd.get("expected_improvement", ""),
                code_changes=hd.get("code_changes", ""),
                success_criteria=hd.get("success_criteria", ""),
                composite=float(hd.get("composite", 0.0)),
            )
        )
    return result


# ------------------------------------------------------------------
# Private reconstruction helpers
# ------------------------------------------------------------------


def _reconstruct_trace(data: dict) -> PipelineTrace:
    """Reconstruct a PipelineTrace from a raw JSON dict.

    Only populates fields needed by ``extract_hypotheses`` — this avoids
    pulling in all 7 module types for reconstruction.
    """
    return PipelineTrace(
        query=data.get("query", ""),
        codebase_path=data.get("codebase_path", ""),
        timestamp=data.get("timestamp", ""),
        hypotheses=data.get("hypotheses", []),
    )
