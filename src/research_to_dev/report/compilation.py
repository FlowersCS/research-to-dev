"""Results compilation — reads per-hypothesis results.tsv files, aggregates
iteration outcomes into HypothesisSummary, computes baseline comparisons,
and produces markdown + JSON reports.

Hexagonal split per D10: domain dataclasses + ResultsReader Protocol in
this module; pure formatting functions (no Protocol); insights.py is a
separate placeholder for future subproblem #2-3.

Usage::

    from research_to_dev.report.compilation import (
        compile_all,
        compile_hypothesis,
        format_markdown,
        format_json,
        TsvResultsReader,
    )

    reader = TsvResultsReader(Path(".research-to-dev/experiments"))
    report = compile_all(reader, reader._experiments_dir)
    print(format_markdown(report))
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TYPE_CHECKING

import typer

from research_to_dev.experiment.program_reader import parse_program_md
from research_to_dev.report.insights import ProgramContext
from research_to_dev.report.traceability import (
    CorrelationTrace,
    HypothesisOrigin,
    TraceabilityContext,
)

if TYPE_CHECKING:
    from research_to_dev.report.insights import InsightsGenerator, InsightsReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TSH header — must match ResultsWriter._COLUMNS exactly
# ---------------------------------------------------------------------------

_TSV_COLUMNS = [
    "iteration",
    "metric_name",
    "metric_value",
    "baseline_value",
    "delta",
    "status",
    "timestamp",
]

_EXPECTED_HEADER = "\t".join(_TSV_COLUMNS)


# ===================================================================
# Domain Dataclasses (D4, D6)
# ===================================================================


@dataclass
class IterationResult:
    """A single iteration row from ``results.tsv``.

    Mirrors the 7-column schema produced by ``ResultsWriter``.
    """

    iteration: int
    metric_name: str
    metric_value: float
    baseline_value: float
    delta: float
    status: str  # "success" | "failed" | "timeout"
    timestamp: str  # ISO 8601


@dataclass
class BaselineComparison:
    """Aggregated improvement summary vs. baseline.

    For maximize direction, higher values are better.
    For minimize direction, lower values are better.
    """

    best_improvement: float | None
    """Most favourable delta among kept iterations.

    For maximize: max(delta). For minimize: min(delta).
    ``None`` if no successful iterations exist."""

    overall_trend: str
    """"improved" | "worsened" | "mixed".

    Only favourable deltas → "improved". Only unfavourable → "worsened".
    Both present → "mixed"."""


@dataclass
class HypothesisSummary:
    """Per-hypothesis compilation of all iteration results."""

    hypothesis_id: str

    status: str
    """"improved" | "worsened" | "crash" | "no_iterations"."""

    iterations_total: int
    iterations_kept: int
    iterations_discarded: int
    iterations_crashed: int

    best_metric_value: float | None
    """Best metric value among successful iterations, judged by direction.
    ``None`` if no successful iterations."""

    baseline_comparison: BaselineComparison | None
    """``None`` if no successful iterations exist."""

    iterations: list[IterationResult] = field(default_factory=list)
    """All iteration rows (including crashed/discarded)."""


@dataclass
class CompiledReport:
    """Top-level report aggregating multiple hypotheses."""

    generated_at: str  # ISO 8601
    hypotheses: list[HypothesisSummary] = field(default_factory=list)
    insights: InsightsReport | None = None
    traceability: TraceabilityContext | None = field(default=None)


# ===================================================================
# ResultsReader Protocol (D8 — Hybrid: Protocol for I/O)
# ===================================================================


class ResultsReader(Protocol):
    """Protocol for reading per-hypothesis iteration results.

    Enables test injection without real filesystem access.
    """

    def read(self, hypothesis_id: str) -> list[IterationResult]:
        """Return all iteration rows for *hypothesis_id*.

        Raises:
            FileNotFoundError: If no results.tsv exists for this hypothesis.
        """
        ...


# ===================================================================
# TSV Results Reader Adapter (RC-06)
# ===================================================================


class TsvResultsReader:
    """Parses ``results.tsv`` files from the experiments directory.

    Constructor-injectable so tests can supply an arbitrary
    ``experiments_dir`` without touching the real filesystem.

    Usage::

        reader = TsvResultsReader(Path(".research-to-dev/experiments"))
        rows = reader.read("abc123")
    """

    def __init__(self, experiments_dir: Path) -> None:
        self._experiments_dir = experiments_dir

    def read(self, hypothesis_id: str) -> list[IterationResult]:
        """Parse ``results.tsv`` for *hypothesis_id*.

        Validates the 7-column header, skips malformed rows (D9),
        and returns ``IterationResult`` objects.

        Raises:
            FileNotFoundError: If the TSV file does not exist.
        """
        tsv_path = self._experiments_dir / hypothesis_id / "results.tsv"

        if not tsv_path.exists():
            raise FileNotFoundError(
                f"results.tsv not found for hypothesis '{hypothesis_id}' "
                f"at {tsv_path}"
            )

        raw = tsv_path.read_text(encoding="utf-8")
        lines = raw.strip().split("\n")

        # Empty file — no header, no data
        if not lines or (len(lines) == 1 and not lines[0].strip()):
            return []

        # Validate header
        if lines[0].strip() != _EXPECTED_HEADER:
            typer.echo(
                f"Warning: Unexpected header in {tsv_path}. "
                f"Expected: {_EXPECTED_HEADER}",
                err=True,
            )

        if len(lines) <= 1:
            # Header only, no data rows
            typer.echo(
                f"Warning: results.tsv for hypothesis '{hypothesis_id}' "
                f"contains no data rows.",
                err=True,
            )
            return []

        results: list[IterationResult] = []
        for line_no, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 7:
                typer.echo(
                    f"Warning: Skipping malformed row at line {line_no} "
                    f"in {tsv_path} (expected 7 columns, got {len(parts)}).",
                    err=True,
                )
                continue

            try:
                results.append(
                    IterationResult(
                        iteration=int(parts[0]),
                        metric_name=parts[1],
                        metric_value=float(parts[2]),
                        baseline_value=float(parts[3]),
                        delta=float(parts[4]),
                        status=parts[5],
                        timestamp=parts[6],
                    )
                )
            except (ValueError, IndexError):
                typer.echo(
                    f"Warning: Skipping malformed row at line {line_no} "
                    f"in {tsv_path} (type coercion failed).",
                    err=True,
                )
                continue

        return results


# ===================================================================
# Compilation Logic (RC-08, RC-09)
# ===================================================================


def compile_hypothesis(
    hypothesis_id: str,
    iterations: list[IterationResult],
    direction: str,
) -> HypothesisSummary:
    """Aggregate raw iterations into a ``HypothesisSummary``.

    Classifies iterations into kept/discarded/crashed based on *direction*
    and delta sign.  Computes best metric value and baseline comparison.

    Args:
        hypothesis_id: Identifier for this hypothesis.
        iterations: All iteration rows (may be empty).
        direction: ``"maximize"`` or ``"minimize"``.

    Returns:
        ``HypothesisSummary`` with status, counts, best metric,
        and baseline comparison.
    """
    total = len(iterations)

    # No iterations edge case
    if total == 0:
        return HypothesisSummary(
            hypothesis_id=hypothesis_id,
            status="no_iterations",
            iterations_total=0,
            iterations_kept=0,
            iterations_discarded=0,
            iterations_crashed=0,
            best_metric_value=None,
            baseline_comparison=None,
            iterations=[],
        )

    # Separate crashed (failed/timeout) from successful
    crashed: list[IterationResult] = []
    successful: list[IterationResult] = []
    for it in iterations:
        if it.status in ("failed", "timeout"):
            crashed.append(it)
        else:
            successful.append(it)

    # Classify successful into kept/discarded by direction
    kept: list[IterationResult] = []
    discarded: list[IterationResult] = []
    for it in successful:
        if direction == "maximize":
            if it.delta >= 0:
                kept.append(it)
            else:
                discarded.append(it)
        else:  # minimize
            if it.delta <= 0:
                kept.append(it)
            else:
                discarded.append(it)

    n_crashed = len(crashed)
    n_kept = len(kept)
    n_discarded = len(discarded)

    # Determine overall status
    if n_kept > 0:
        status = "improved"
    elif len(successful) == 0:
        status = "crash"
    else:
        status = "worsened"

    # Best metric value among successful (by direction)
    if successful:
        if direction == "maximize":
            best_metric_value = max(it.metric_value for it in successful)
        else:
            best_metric_value = min(it.metric_value for it in successful)
    else:
        best_metric_value = None

    # Baseline comparison
    if successful:
        # Best improvement: most favourable delta by direction
        if direction == "maximize":
            best_improvement = max(it.delta for it in successful)
        else:
            best_improvement = min(it.delta for it in successful)

        # Overall trend
        favourable = 0
        unfavourable = 0
        for it in successful:
            if direction == "maximize":
                if it.delta >= 0:
                    favourable += 1
                else:
                    unfavourable += 1
            else:  # minimize
                if it.delta <= 0:
                    favourable += 1
                else:
                    unfavourable += 1

        if favourable > 0 and unfavourable > 0:
            overall_trend = "mixed"
        elif favourable > 0:
            overall_trend = "improved"
        else:
            overall_trend = "worsened"

        baseline_comparison = BaselineComparison(
            best_improvement=best_improvement,
            overall_trend=overall_trend,
        )
    else:
        baseline_comparison = None

    return HypothesisSummary(
        hypothesis_id=hypothesis_id,
        status=status,
        iterations_total=total,
        iterations_kept=n_kept,
        iterations_discarded=n_discarded,
        iterations_crashed=n_crashed,
        best_metric_value=best_metric_value,
        baseline_comparison=baseline_comparison,
        iterations=iterations,
    )


def compile_all(
    reader: ResultsReader,
    experiments_dir: Path,
    insights_generator: InsightsGenerator | None = None,
    traceability: TraceabilityContext | None = None,
) -> CompiledReport:
    """Compile all experiment results into a ``CompiledReport``.

    Iterates subdirectories of *experiments_dir*, reads per-hypothesis
    ``results.tsv`` via the *reader* Protocol, discovers direction from
    ``program.md`` (falls back to ``"maximize"``).

    When *insights_generator* is provided, collects ``ProgramContext``
    from each hypothesis's ``program.md`` and generates cross-hypothesis
    insights after compilation.  Insights generation failure is graceful
    — the report compiles without insights rather than crashing.

    Hypothesis directories that lack a ``results.tsv`` are silently
    skipped (``FileNotFoundError`` caught).  An empty experiments
    directory yields an empty ``CompiledReport``.

    Args:
        reader: A ``ResultsReader`` implementation.
        experiments_dir: Path to the experiments directory containing
            per-hypothesis subdirectories.
        insights_generator: Optional ``InsightsGenerator`` for
            cross-hypothesis LLM analysis.  When ``None`` (default),
            no insights are generated.

    Returns:
        ``CompiledReport`` with one ``HypothesisSummary`` per experiment
        directory that has results.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    hypotheses: list[HypothesisSummary] = []
    program_contexts: list[ProgramContext] = []

    if not experiments_dir.exists() or not experiments_dir.is_dir():
        return CompiledReport(generated_at=generated_at, hypotheses=[])

    for entry in sorted(experiments_dir.iterdir()):
        if not entry.is_dir():
            continue

        hypothesis_id = entry.name

        # --- read iterations via Protocol --------------------------------
        try:
            iterations = reader.read(hypothesis_id)
        except FileNotFoundError:
            continue

        # --- discover direction from program.md ---------------------------
        direction = "maximize"
        program_spec = None
        program_md_path = entry / "program.md"
        if program_md_path.exists():
            try:
                program_spec = parse_program_md(program_md_path)
                direction = program_spec.direction
            except (ValueError, FileNotFoundError):
                # Malformed program.md — fall back to default
                pass

        # --- collect ProgramContext for insights enrichment ---------------
        if program_spec is not None:
            baseline_str = ", ".join(
                f"{k}={v}" for k, v in program_spec.baseline.items()
            )
            program_contexts.append(
                ProgramContext(
                    hypothesis_id=hypothesis_id,
                    direction=program_spec.direction,
                    baseline=baseline_str,
                    success_criteria=program_spec.success_criteria,
                )
            )

        summary = compile_hypothesis(hypothesis_id, iterations, direction)
        hypotheses.append(summary)

    # --- generate cross-hypothesis insights --------------------------------
    insights: InsightsReport | None = None
    if insights_generator is not None and hypotheses:
        try:
            insights = asyncio.run(
                insights_generator.generate(hypotheses, program_contexts)
            )
        except Exception as exc:
            logger.warning("Insights generation failed: %s", exc)

    return CompiledReport(
        generated_at=generated_at,
        hypotheses=hypotheses,
        insights=insights,
        traceability=traceability,
    )


# ===================================================================
# Formatters — pure functions (D8: no Protocol needed)
# ===================================================================


def format_markdown(report: CompiledReport) -> str:
    """Render a ``CompiledReport`` as a human-readable Markdown string.

    Produces one section per hypothesis with status, counts, best metric,
    baseline comparison, and an iteration table.
    """
    lines: list[str] = []
    lines.append(f"# Experiment Results — {report.generated_at}")
    lines.append("")

    if not report.hypotheses:
        lines.append("*No hypothesis results to report.*")
        lines.append("")
        return "\n".join(lines)

    for hyp in report.hypotheses:
        lines.append(f'<a id="hyp-{hyp.hypothesis_id}"></a>')
        lines.append("")
        lines.append(f"## Hypothesis: {hyp.hypothesis_id}")
        lines.append("")

        # Status line
        lines.append(f"- **Status**: {hyp.status}")

        # Iteration counts
        parts = [f"{hyp.iterations_total} total"]
        if hyp.iterations_kept:
            parts.append(f"{hyp.iterations_kept} kept")
        if hyp.iterations_discarded:
            parts.append(f"{hyp.iterations_discarded} discarded")
        if hyp.iterations_crashed:
            parts.append(f"{hyp.iterations_crashed} crashed")
        lines.append(f"- **Iterations**: {', '.join(parts)}")

        # Best metric
        if hyp.best_metric_value is not None:
            # Find the iteration that matches best_metric_value to get its delta
            best_delta: float | None = None
            for it in hyp.iterations:
                if it.metric_value == hyp.best_metric_value:
                    best_delta = it.delta
                    break
            delta_str = f" (delta: {best_delta:+.4f})" if best_delta is not None else ""
            metric_name = hyp.iterations[0].metric_name if hyp.iterations else "metric"
            lines.append(
                f"- **Best metric**: {metric_name}={hyp.best_metric_value}{delta_str}"
            )
        else:
            lines.append("- **Best metric**: N/A")

        # Baseline comparison
        if hyp.baseline_comparison is not None:
            bc = hyp.baseline_comparison
            imp_str = f"{bc.best_improvement:+.4f}" if bc.best_improvement is not None else "N/A"
            lines.append(
                f"- **Baseline comparison**: {bc.overall_trend} "
                f"(best improvement: {imp_str})"
            )
        else:
            lines.append("- **Baseline comparison**: N/A (no successful iterations)")

        lines.append("")

        # Iteration table
        if hyp.iterations:
            lines.append("| # | Metric | Value | Delta | Status |")
            lines.append("|---|--------|-------|-------|--------|")
            for it in hyp.iterations:
                lines.append(
                    f"| {it.iteration} "
                    f"| {it.metric_name} "
                    f"| {it.metric_value} "
                    f"| {it.delta:+.4f} "
                    f"| {it.status} |"
                )
            lines.append("")
        else:
            lines.append("*No iterations recorded.*")
            lines.append("")

        # --- Traceability subsection (TR-12) ---
        if report.traceability is not None:
            origin = report.traceability.hypothesis_origins.get(hyp.hypothesis_id)
            if origin is not None:
                lines.append("### Traceability")
                lines.append("")
                lines.append(f"- **Title**: {origin.title or 'N/A'}")
                # Description truncated to 80 chars
                desc = origin.description or ""
                if len(desc) > 80:
                    desc = desc[:80] + "…"
                lines.append(f"- **Description**: {desc}")
                # Supporting papers
                if origin.supporting_papers:
                    lines.append(
                        f"- **Supporting papers**: "
                        f"{', '.join(origin.supporting_papers)}"
                    )
                else:
                    lines.append("- **Supporting papers**: None")
                lines.append("")

                # Resolved correlations
                if origin.resolved_correlations:
                    lines.append("#### Resolved Correlations")
                    lines.append("")
                    for ct in origin.resolved_correlations:
                        # Truncate claim text for display
                        claim_short = (
                            ct.claim_text[:120] + "…"
                            if len(ct.claim_text) > 120
                            else ct.claim_text
                        )
                        # Reference to claim/component in appendix
                        claim_anchor = report.traceability.claim_anchors
                        comp_anchor = report.traceability.component_anchors
                        # Find the matching claim anchor
                        claim_link = ""
                        for aid, atext in claim_anchor.items():
                            if atext == ct.claim_text:
                                claim_link = f"[claim](#{aid})"
                                break
                        comp_link = ""
                        for aid, (fp, mn, sig) in comp_anchor.items():
                            if mn == ct.component_module_name:
                                comp_link = f"[component](#{aid})"
                                break

                        lines.append(f"- **{ct.correlation_type}**: {claim_short}")
                        if claim_link or comp_link:
                            links = " · ".join(
                                l for l in [claim_link, comp_link] if l
                            )
                            lines.append(f"  📎 {links}")
                        lines.append("")
                else:
                    if origin.unresolved_correlation_ids:
                        lines.append("#### Resolved Correlations")
                        lines.append("")
                    lines.append("No correlations resolved")
                    lines.append("")

                # Unresolved correlations
                if origin.unresolved_correlation_ids:
                    lines.append("#### Unresolved")
                    lines.append("")
                    for uid in origin.unresolved_correlation_ids:
                        lines.append(f"- ⚠ Unresolved: {uid}")
                    lines.append("")

                lines.append(
                    "[Full traceability detail →](#traceability-appendix)"
                )
                lines.append("")

    # --- Cross-Hypothesis Insights section (if available) ---
    if report.insights is not None:
        from research_to_dev.report.insights import InsightsReport  # runtime import

        insights: InsightsReport = report.insights

        lines.append("---")
        lines.append("")
        lines.append("## Cross-Hypothesis Insights")
        lines.append("")

        # -- Patterns --
        lines.append("### Patterns")
        lines.append("")
        if insights.patterns:
            for p in insights.patterns:
                lines.append(f"**{p.title}** [{p.confidence}]")
                lines.append(p.description)
                lines.append("")
                lines.append(
                    f"Hypotheses: {', '.join(p.affected_hypotheses)}"
                )
                lines.append("")
        else:
            lines.append("No patterns detected across hypotheses.")
            lines.append("")

        # -- Recommendations --
        lines.append("### Recommendations")
        lines.append("")
        if insights.recommendations:
            for r in insights.recommendations:
                lines.append(f"- **[{r.priority}]** {r.action}")
                lines.append(f"  {r.rationale}")
                lines.append(
                    f"  Evidence: {', '.join(r.supporting_evidence)}"
                )
                lines.append("")
        else:
            lines.append("No recommendations.")
            lines.append("")

        # -- Evidence Correlations --
        lines.append("### Evidence Correlations")
        lines.append("")
        if insights.evidence_correlation:
            for ec in insights.evidence_correlation:
                lines.append(f"**{ec.paper_id}**")
                lines.append(ec.evidence_summary)
                lines.append(
                    f"Hypotheses: {', '.join(ec.related_hypotheses)}"
                    f" · Code: {', '.join(ec.code_areas)}"
                )
                lines.append("")
        else:
            lines.append("No evidence correlations found.")
            lines.append("")

    # --- Traceability Appendix (TR-13) ---
    if report.traceability is not None:
        lines.append("---")
        lines.append("")
        lines.append("## Traceability Appendix")
        lines.append("")
        lines.append('<a id="traceability-appendix"></a>')
        lines.append("")

        # -- Claims section --
        lines.append("### Claims")
        lines.append("")
        if report.traceability.claim_anchors:
            for anchor_id, claim_text in report.traceability.claim_anchors.items():
                lines.append(f'<a id="{anchor_id}"></a>')
                lines.append("")
                lines.append(f"**Claim `{anchor_id}`**")
                lines.append("")
                lines.append(claim_text)
                lines.append("")
                # List components mapped to this claim
                # We need to find which components reference this claim
                # For now, list all components (they are independently rendered)
                lines.append("")
        else:
            lines.append("*No claims recorded.*")
            lines.append("")

        # -- Code Components section --
        lines.append("### Code Components")
        lines.append("")
        if report.traceability.component_anchors:
            for anchor_id, (file_path, module_name, signature) in report.traceability.component_anchors.items():
                lines.append(f'<a id="{anchor_id}"></a>')
                lines.append("")
                lines.append(f"**Component `{anchor_id}`**")
                lines.append("")
                lines.append(f"- **File**: `{file_path}`" if file_path else "- **File**: N/A")
                lines.append(f"- **Module**: `{module_name}`")
                lines.append(f"- **Signature**: `{signature}`")
                lines.append("")
        else:
            lines.append("*No code components recorded.*")
            lines.append("")

    return "\n".join(lines)


def _insights_to_dict(insights) -> dict:
    """Convert an InsightsReport to a JSON-serializable dict."""
    # InsightsReport imported at runtime (TYPE_CHECKING guard at top)
    from research_to_dev.report.insights import InsightsReport  # noqa: F401

    return {
        "patterns": [
            {
                "title": p.title,
                "description": p.description,
                "affected_hypotheses": p.affected_hypotheses,
                "confidence": p.confidence,
            }
            for p in insights.patterns
        ],
        "recommendations": [
            {
                "action": r.action,
                "priority": r.priority,
                "rationale": r.rationale,
                "supporting_evidence": r.supporting_evidence,
            }
            for r in insights.recommendations
        ],
        "evidence_correlation": [
            {
                "paper_id": ec.paper_id,
                "evidence_summary": ec.evidence_summary,
                "related_hypotheses": ec.related_hypotheses,
                "code_areas": ec.code_areas,
            }
            for ec in insights.evidence_correlation
        ],
    }


def _traceability_to_dict(traceability: TraceabilityContext) -> dict:
    """Convert a TraceabilityContext to a JSON-serializable dict."""

    def _correlation_to_dict(ct: CorrelationTrace) -> dict:
        return {
            "correlation_id": ct.correlation_id,
            "paper_title": ct.paper_title,
            "claim_text": ct.claim_text,
            "claim_section": ct.claim_section,
            "claim_paper_id": ct.claim_paper_id,
            "component_name": ct.component_name,
            "component_file_path": ct.component_file_path,
            "component_module_name": ct.component_module_name,
            "component_signature": ct.component_signature,
            "correlation_type": ct.correlation_type,
            "reasoning": ct.reasoning,
            "target_type": ct.target_type,
        }

    def _origin_to_dict(origin: HypothesisOrigin) -> dict:
        return {
            "hypothesis_id": origin.hypothesis_id,
            "title": origin.title,
            "description": origin.description,
            "code_changes": origin.code_changes,
            "supporting_papers": origin.supporting_papers,
            "resolved_correlations": [
                _correlation_to_dict(ct)
                for ct in origin.resolved_correlations
            ],
            "unresolved_correlation_ids": origin.unresolved_correlation_ids,
        }

    return {
        "hypothesis_origins": {
            hid: _origin_to_dict(origin)
            for hid, origin in traceability.hypothesis_origins.items()
        },
        "claim_anchors": dict(traceability.claim_anchors),
        "component_anchors": {
            aid: list(data)
            for aid, data in traceability.component_anchors.items()
        },
    }


def format_json(report: CompiledReport) -> str:
    """Render a ``CompiledReport`` as an indented JSON string.

    Serialises ``generated_at`` and full ``hypotheses`` array with
    per-hypothesis iteration detail.
    """

    def _hyp_to_dict(hyp: HypothesisSummary) -> dict:
        result: dict = {
            "hypothesis_id": hyp.hypothesis_id,
            "status": hyp.status,
            "iterations_total": hyp.iterations_total,
            "iterations_kept": hyp.iterations_kept,
            "iterations_discarded": hyp.iterations_discarded,
            "iterations_crashed": hyp.iterations_crashed,
            "best_metric_value": hyp.best_metric_value,
            "baseline_comparison": (
                {
                    "best_improvement": hyp.baseline_comparison.best_improvement,
                    "overall_trend": hyp.baseline_comparison.overall_trend,
                }
                if hyp.baseline_comparison is not None
                else None
            ),
            "iterations": [
                {
                    "iteration": it.iteration,
                    "metric_name": it.metric_name,
                    "metric_value": it.metric_value,
                    "baseline_value": it.baseline_value,
                    "delta": it.delta,
                    "status": it.status,
                    "timestamp": it.timestamp,
                }
                for it in hyp.iterations
            ],
        }
        return result

    result_dict: dict = {
        "generated_at": report.generated_at,
        "hypotheses": [_hyp_to_dict(h) for h in report.hypotheses],
    }

    if report.insights is not None:
        result_dict["insights"] = _insights_to_dict(report.insights)

    if report.traceability is not None:
        result_dict["traceability"] = _traceability_to_dict(report.traceability)

    return json.dumps(result_dict, indent=2)


# ===================================================================
# Report Writer (RC-15)
# ===================================================================


def write_reports(report: CompiledReport, reports_dir: Path) -> tuple[Path, Path]:
    """Write ``.md`` and ``.json`` report files to *reports_dir*.

    Generates timestamped filenames (e.g.
    ``report-2026-06-07T143000Z.md``).  Creates *reports_dir* and
    any necessary parent directories automatically.

    Args:
        report: The compiled report to write.
        reports_dir: Destination directory.

    Returns:
        Tuple of ``(md_path, json_path)`` for the written files.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Parse generated_at back to datetime for clean filename slug
    dt = datetime.fromisoformat(report.generated_at)
    ts_slug = dt.strftime("%Y%m%d-%H%M%S")

    md_path = reports_dir / f"report-{ts_slug}.md"
    json_path = reports_dir / f"report-{ts_slug}.json"

    md_path.write_text(format_markdown(report), encoding="utf-8")
    json_path.write_text(format_json(report), encoding="utf-8")

    return md_path, json_path
