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
from research_to_dev.experiment.git import GitOperations
from research_to_dev.experiment.metrics import MetricRegistry
from research_to_dev.experiment.program import ProgramWriter
from research_to_dev.experiment.setup import ExperimentSetup
from research_to_dev.experiment.trace import extract_hypotheses, read_trace
from research_to_dev.shared.config import ExperimentConfig

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


def _persist_trace(trace: PipelineTrace) -> None:
    """Side-effect: persist PipelineTrace to canonical location (Gap 1).

    Writes to ``.research-to-dev/pipeline/trace.json``.  Creates the
    parent directory if it doesn't exist.  On failure, emits a warning
    but does NOT abort — the primary output is the user-specified file
    (ES-03C).
    """
    canonical = Path(".research-to-dev/pipeline/trace.json")
    try:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(trace.to_json(), encoding="utf-8")
    except OSError:
        typer.echo(
            f"Warning: Could not write trace to {canonical}",
            err=True,
        )


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

    # -- 4b. Trace persistence side-effect (Gap 1) ---------------------
    _persist_trace(trace)

    # -- 5. Terminal summary -------------------------------------------
    elapsed = time.monotonic() - t_start
    _print_summary(trace, output, elapsed)

    # -- 6. Exit code: 1 if retrieval was empty (fail-fast) ------------
    retrieval_step = trace.steps.get("retrieval")
    if retrieval_step and retrieval_step.status == "empty":
        raise typer.Exit(code=1)


# ======================================================================
# Config command group (D11, ES-02)
# ======================================================================

config_app = typer.Typer(
    help="Manage .research-to-dev/config.yaml for custom metric patterns.",
    no_args_is_help=True,
)


@config_app.command(name="init")
def config_init(
    path: str = typer.Option(
        ".research-to-dev/config.yaml",
        "--path",
        "-p",
        help="Path to write the config file (default: .research-to-dev/config.yaml).",
    ),
) -> None:
    """Initialize a project config with documented built-in metric patterns.

    Generates ``.research-to-dev/config.yaml`` with all built-in metric
    patterns documented and commented-out examples for custom metrics.

    The command is idempotent — it fails loud if the file already exists
    (ES-02B).
    """
    from research_to_dev.experiment.metrics import MetricRegistry

    config_file = Path(path)
    if config_file.exists():
        typer.echo(
            f"Error: Config already exists at {path}",
            err=True,
        )
        raise typer.Exit(code=1)

    registry = MetricRegistry()

    lines: list[str] = [
        "# Research-to-Dev configuration file.",
        "# Generated by `research-to-dev config init`.",
        "#",
        "# This file defines per-project overrides for metric extraction patterns.",
        "# Built-in metrics are listed below (commented out). Uncomment and",
        "# modify patterns to customize, or add your own custom metrics.",
        "#",
        "# Regex patterns use Python's `re` module syntax.",
        "# Example: r\"accuracy[:\\s=]*([\\d.]+)\" extracts a float after 'accuracy: '.",
        "",
        "metrics:",
    ]

    # Add built-in metrics as commented examples
    configs = registry.to_configs()
    for mc in configs:
        lines.append(f"  # {mc.name}: \"{mc.pattern}\"  # built-in default")

    lines.extend([
        "",
        "  # Custom metric examples (uncomment and modify):",
        "  # custom_auc: \"auc[:\\s=]*([\\d.]+)\"",
    ])

    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    typer.echo(f"Created {path}")


# ======================================================================
# Experiment command group (D1, D2, D8, ES-04)
# ======================================================================

experiment_app = typer.Typer(
    help="Set up and manage isolated experiment environments.",
    no_args_is_help=True,
)


@experiment_app.command(name="setup")
def experiment_setup(
    hypothesis_id: str = typer.Option(
        ...,
        "--hypothesis-id",
        help="SHA-256 ID of the hypothesis to experiment on (required).",
    ),
    time_budget: str = typer.Option(
        ...,
        "--time-budget",
        help="Time budget for the experiment (e.g. '30m', '2h', '1h30m').",
    ),
    max_iterations: int = typer.Option(
        ...,
        "--max-iterations",
        help="Maximum number of improvement iterations (>= 1).",
    ),
    trace: str = typer.Option(
        ".research-to-dev/pipeline/trace.json",
        "--trace",
        "-t",
        help="Path to the pipeline trace JSON (default: .research-to-dev/pipeline/trace.json).",
    ),
    config_path: str = typer.Option(
        ".research-to-dev/config.yaml",
        "--config",
        help="Path to config.yaml for custom metric patterns.",
    ),
) -> None:
    """Set up an isolated experiment environment for a hypothesis.

    Reads the hypothesis from the pipeline trace, validates the target
    metric, creates a git branch ``experiment/<id>``, and generates
    ``program.md`` with experiment instructions.
    """
    # -- 1. Load trace and find hypothesis -----------------------------
    try:
        pipeline_trace = read_trace(trace)
    except FileNotFoundError:
        typer.echo(
            f"Error: Trace not found at {trace}. "
            f"Run `research-to-dev analyze` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    hypotheses = extract_hypotheses(pipeline_trace)
    if not hypotheses:
        typer.echo(
            "Error: No hypotheses found in trace. "
            "Run `research-to-dev analyze` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Find the target hypothesis by ID
    target = None
    for h in hypotheses:
        if h.id == hypothesis_id:
            target = h
            break

    if target is None:
        typer.echo(
            f"Error: Hypothesis '{hypothesis_id}' not found in trace.",
            err=True,
        )
        raise typer.Exit(code=1)

    # -- 2. Validate target metric --------------------------------------
    metrics = MetricRegistry(config_path=config_path if Path(config_path).exists() else None)
    target_metric = target.target_metric

    if not metrics.validate_metric(target_metric):
        builtins = ", ".join(metrics.list_all())
        typer.echo(
            f"Error: Target metric '{target_metric}' not found in "
            f"built-in metrics or config.yaml.\n"
            f"Available built-in metrics: {builtins}\n"
            f"Define custom metrics via `research-to-dev config init` "
            f"and edit .research-to-dev/config.yaml",
            err=True,
        )
        raise typer.Exit(code=1)

    # -- 3. Build config and run setup ----------------------------------
    config = ExperimentConfig(
        time_budget=time_budget,
        max_iterations=max_iterations,
        trace_path=trace,
        config_path=config_path,
        hypothesis_id=hypothesis_id,
    )

    git_ops = GitOperations(repo_path=".")
    writer = ProgramWriter(base_dir=".research-to-dev/experiments")
    setup = ExperimentSetup(git_ops=git_ops, writer=writer, metrics=metrics)

    try:
        program_path = setup.run(target, config, target_metric)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Experiment branch created: experiment/{hypothesis_id}")
    typer.echo(f"Program.md generated: {program_path}")


# -- Register sub-apps on the main app ------------------------------------

app.add_typer(config_app, name="config")
app.add_typer(experiment_app, name="experiment")


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
