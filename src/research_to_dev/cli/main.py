"""CLI entry points for the research-to-dev tool.

Uses ``typer`` — top-level ``analyze`` command that runs the full
7-step research-to-code pipeline.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI

from research_to_dev.cli.orchestrator import PipelineOrchestrator, PipelineTrace

app = typer.Typer(
    name="research-to-dev",
    help="Research-to-development pipeline: find papers, analyze code, generate hypotheses.",
    no_args_is_help=True,
)


@app.callback()
def _app_callback() -> None:
    """Entry point for research-to-dev. Use subcommands like 'analyze'."""


def _on_step(msg: str, step: int) -> None:
    """Progress callback bridging orchestrator events to typer output."""
    if step == 0:
        typer.echo(f"\n  {msg}")
    else:
        typer.echo(f"  >>> Step {step}/7: {msg}")


# ======================================================================
# Analyze command
# ======================================================================


@app.command()
def analyze(
    query: str = typer.Option(
        ...,
        "--query",
        "-q",
        help="Research question to explore.",
    ),
    codebase: str = typer.Option(
        ".",
        "--codebase",
        "-c",
        help="Path to the codebase directory (default: current directory).",
    ),
    output: str = typer.Option(
        "results.json",
        "--output",
        "-o",
        help="JSON output file path (default: results.json).",
    ),
) -> None:
    """Run the full research-to-code pipeline.

    Searches academic sources, extracts content, ranks by relevance,
    profiles papers, analyzes the target codebase, correlates claims
    with code, and generates actionable improvement hypotheses.
    """
    # -- 0. Load environment -------------------------------------------
    load_dotenv()

    # -- 1. Validate OPENAI_API_KEY ------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.echo(
            "Error: OPENAI_API_KEY not found.\n"
            "  Set it via environment variable or .env file.",
            err=True,
        )
        raise typer.Exit(code=1)

    # -- 2. Validate codebase path -------------------------------------
    codebase_path = Path(codebase).resolve()
    if not codebase_path.is_dir():
        typer.echo(
            f"Error: '{codebase}' is not a valid directory.",
            err=True,
        )
        raise typer.Exit(code=1)

    # -- 3. Run pipeline -----------------------------------------------
    t_start = time.monotonic()

    client = AsyncOpenAI(api_key=api_key)
    orchestrator = PipelineOrchestrator(client, on_step=_on_step)

    typer.echo(f"\n  Query:     \"{query}\"")
    typer.echo(f"  Codebase:  {codebase_path}")
    typer.echo(f"  Output:    {output}\n")

    trace = asyncio.run(orchestrator.run(query, str(codebase_path)))

    # -- 4. Write output file ------------------------------------------
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(trace.to_json(), encoding="utf-8")

    # -- 5. Terminal summary -------------------------------------------
    elapsed = time.monotonic() - t_start
    _print_summary(trace, output, elapsed)

    # -- 6. Exit code: 1 if retrieval was empty (fail-fast) ------------
    retrieval_step = trace.steps.get("retrieval")
    if retrieval_step and retrieval_step.status == "empty":
        raise typer.Exit(code=1)


# ======================================================================
# Terminal summary rendering
# ======================================================================


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis if longer than max_len."""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _print_summary(
    trace: PipelineTrace, output_path: str, elapsed: float
) -> None:
    """Print a human-readable summary after pipeline completion."""

    typer.echo()
    typer.echo("═══════════════════════════════════════")
    typer.echo(" PIPELINE RESULTS")
    typer.echo("═══════════════════════════════════════")

    # Per-step counts
    retrieval_count = len(trace.retrieval.results) if trace.retrieval else 0
    ranking_count = len(trace.ranking.papers) if trace.ranking else 0
    codebase_modules = (
        len(trace.codebase.modules) if trace.codebase else 0
    )
    correlations_count = 0
    if trace.correlation:
        correlations_count = (
            len(trace.correlation.correlations)
            + len(trace.correlation.module_correlations)
        )

    typer.echo(
        f"Query: \"{trace.query}\" | "
        f"Papers: {retrieval_count} ranked | "
        f"Codebase: {codebase_modules} modules | "
        f"Correlations: {correlations_count}"
    )
    typer.echo()

    # Top hypotheses
    hypo_step = trace.steps.get("hypothesis")
    if trace.hypotheses:
        typer.echo(" Top Hypotheses:")
        typer.echo(f" {'#':>3}  {'Score':>5}  Hypothesis")
        for i, h in enumerate(trace.hypotheses[:5], 1):
            title = _truncate(h["title"], 50)
            score = h["composite"]
            typer.echo(f" {i:>3}  {score:>5.1f}  {title}")
    elif hypo_step and hypo_step.status == "success":
        typer.echo(" No hypotheses generated (pipeline completed but produced none).")
    elif trace.hypothesis is not None:
        typer.echo(" No hypotheses generated.")

    typer.echo()

    # Footer
    typer.echo(f" Output:   {output_path}")
    n_warnings = len(trace.warnings)
    if n_warnings > 0:
        typer.echo(f" Warnings: {n_warnings}")
        for w in trace.warnings[:3]:
            typer.echo(f"   - {_truncate(w, 80)}")
        if n_warnings > 3:
            typer.echo(f"   ... and {n_warnings - 3} more")
    typer.echo(f" Time:     {elapsed:.1f}s")
    typer.echo("═══════════════════════════════════════")
