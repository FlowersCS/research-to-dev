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
from research_to_dev.experiment.agent import OpenCodeAdapter
from research_to_dev.experiment.executor import CodeExecutor
from research_to_dev.experiment.git import GitOperations
from research_to_dev.experiment.metrics import MetricRegistry
from research_to_dev.experiment.program import ProgramWriter
from research_to_dev.experiment.program_reader import parse_program_md
from research_to_dev.experiment.results import ResultsWriter
from research_to_dev.experiment.runner import ExperimentRunner
from research_to_dev.experiment.setup import ExperimentSetup
from research_to_dev.experiment.trace import extract_hypotheses, read_trace
from research_to_dev.shared.config import ExperimentConfig, ReportConfig

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
    run_command: str = typer.Option(
        ...,
        "--run-command",
        help="Shell command to execute (e.g. 'python train.py').",
    ),
    baseline: str = typer.Option(
        ...,
        "--baseline",
        help="Baseline metric as key=value (e.g. 'val_accuracy=0.72').",
    ),
    coding_agent_model: str = typer.Option(
        ...,
        "--coding-agent-model",
        help="Model identifier for the coding agent (e.g. 'opencode-go/deepseek-v4-pro').",
    ),
    direction: str | None = typer.Option(
        None,
        "--direction",
        help="Explicit direction override: 'maximize' or 'minimize'. "
             "If not set, inferred from success_criteria.",
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

    # -- 3. Parse baseline from key=value format ------------------------
    if "=" not in baseline:
        typer.echo(
            f"Error: baseline must be in key=value format "
            f"(e.g. 'val_accuracy=0.72'), got '{baseline}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    key, _, val_str = baseline.partition("=")
    key = key.strip()
    val_str = val_str.strip()

    if not key or not val_str:
        typer.echo(
            f"Error: baseline must be in key=value format "
            f"(e.g. 'val_accuracy=0.72'), got '{baseline}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        baseline_dict: dict[str, float] = {key: float(val_str)}
    except ValueError:
        typer.echo(
            f"Error: baseline value must be numeric (e.g. '0.72'), "
            f"got '{val_str}'.",
            err=True,
        )
        raise typer.Exit(code=1)

    # -- 4. Build config and run setup ----------------------------------
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
        program_path = setup.run(
            target,
            config,
            target_metric,
            run_command=run_command,
            baseline=baseline_dict,
            coding_agent_model=coding_agent_model,
            direction=direction,
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Experiment branch created: experiment/{hypothesis_id}")
    typer.echo(f"Program.md generated: {program_path}")


@experiment_app.command(name="run")
def experiment_run(
    hypothesis_id: str = typer.Argument(
        ...,
        help="SHA-256 ID of the hypothesis to experiment on.",
    ),
) -> None:
    """Run the experiment iteration loop for a hypothesis.

    Reads program.md from .research-to-dev/experiments/<hypothesis-id>/,
    instantiates the full dependency chain, and executes the agentic
    iteration loop until the time budget or max iterations is reached.

    All configuration (time budget, max iterations, metric direction, etc.)
    comes from program.md — there are no CLI overrides (D11).
    """
    # -- 1. Resolve and validate program.md path -------------------------
    program_md_path = Path(
        f".research-to-dev/experiments/{hypothesis_id}/program.md"
    ).resolve()

    if not program_md_path.exists():
        typer.echo(
            f"Error: program.md not found at {program_md_path}.\n"
            f"Run 'research-to-dev experiment setup --hypothesis-id "
            f"{hypothesis_id} ...' first.",
            err=True,
        )
        raise typer.Exit(code=1)

    # -- 2. Parse program.md (AE-25 step 2) ------------------------------
    try:
        program_spec = parse_program_md(program_md_path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    # -- 3. Load config.yaml (optional, AE-25 step 3) --------------------
    config_yaml_path = ".research-to-dev/config.yaml"
    registry = MetricRegistry(
        config_path=config_yaml_path if Path(config_yaml_path).exists() else None
    )

    # -- 4-9. Build dependency chain (AE-25 steps 4-9) -------------------
    git_ops = GitOperations(repo_path=".")
    adapter = OpenCodeAdapter()
    executor = CodeExecutor(cwd=".")
    results_path = Path(
        f".research-to-dev/experiments/{hypothesis_id}/results.tsv"
    )
    writer = ResultsWriter(tsv_path=results_path)

    runner_obj = ExperimentRunner(
        agent=adapter,
        executor=executor,
        git_ops=git_ops,
        metric_registry=registry,
        results_writer=writer,
        program_spec=program_spec,
    )

    # -- 10. Run the experiment loop (AE-25 step 10) ---------------------
    try:
        runner_obj.run(program_md_path)
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    # -- 11. Print completion summary (AE-26) ----------------------------
    _print_run_summary(results_path, program_spec)


# -- Register sub-apps on the main app ------------------------------------

app.add_typer(config_app, name="config")
app.add_typer(experiment_app, name="experiment")


# ======================================================================
# Report command (D1, D2, D3, RC-17 through RC-19)
# ======================================================================


@app.command(name="report")
def report(
    hypothesis: str | None = typer.Option(
        None,
        "--hypothesis",
        help="Compile report for a specific hypothesis ID only.",
    ),
    with_insights: bool = typer.Option(
        False,
        "--with-insights",
        help="Include LLM-generated insights in the report.",
    ),
    trace: str = typer.Option(
        "",
        "--trace",
        "-t",
        help="Path to pipeline trace.json for traceability links.",
    ),
) -> None:
    """Compile experiment results into markdown and JSON reports.

    Reads per-hypothesis ``results.tsv`` from
    ``.research-to-dev/experiments/``, aggregates summaries, and writes
    ``.md`` + ``.json`` reports to ``.research-to-dev/reports/``.

    By default, all hypotheses are included.  Use ``--hypothesis <id>``
    to filter to a single one.
    """
    from research_to_dev.report.compilation import (
        CompiledReport,
        TsvResultsReader,
        compile_all,
        compile_hypothesis,
        write_reports,
    )

    t_start = time.monotonic()

    config = ReportConfig(
        hypothesis_filter=hypothesis,
        include_insights=with_insights,
    )

    # -- Insights generator wiring ----------------------------------------
    insights_generator = None
    if config.include_insights:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            typer.echo(
                "Warning: OPENAI_API_KEY not set. "
                "Insights generation skipped.",
                err=True,
            )
        else:
            from research_to_dev.report.insights import OpenAIInsightsGenerator

            insights_client = AsyncOpenAI(api_key=api_key)
            insights_generator = OpenAIInsightsGenerator(
                client=insights_client,
                model=config.insights_model,
            )

    experiments_dir = Path(config.experiments_dir)
    reports_dir = Path(config.reports_dir)

    # -- 0. Traceability wiring (TR-16) -----------------------------------
    trace_path = trace or ".research-to-dev/pipeline/trace.json"
    traceability_ctx = None
    if trace_path:
        try:
            from research_to_dev.report.traceability import JsonTraceReader

            reader_trace = JsonTraceReader(trace_path)
            traceability_ctx = reader_trace.read()
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.echo(
                f"Warning: Could not read trace at {trace_path}: {exc}. "
                f"Report compiled without traceability.",
                err=True,
            )

    # -- 1. Validate experiments directory --------------------------------
    if not experiments_dir.exists() or not experiments_dir.is_dir():
        typer.echo(
            f"Error: Experiments directory not found at {experiments_dir}. "
            f"Run `research-to-dev experiment setup ...` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    # -- 2. Build reader --------------------------------------------------
    reader = TsvResultsReader(experiments_dir)

    # -- 3. Compile (all or single) ---------------------------------------
    if config.hypothesis_filter:
        # Single hypothesis mode
        hyp_dir = experiments_dir / config.hypothesis_filter
        if not hyp_dir.is_dir():
            typer.echo(
                f"Error: Hypothesis directory not found: {hyp_dir}",
                err=True,
            )
            raise typer.Exit(code=1)

        try:
            iterations = reader.read(config.hypothesis_filter)
        except FileNotFoundError:
            typer.echo(
                f"Error: results.tsv not found for hypothesis "
                f"'{config.hypothesis_filter}'.",
                err=True,
            )
            raise typer.Exit(code=1)

        if not iterations:
            typer.echo(
                f"Warning: Empty results for hypothesis "
                f"'{config.hypothesis_filter}' (no iterations).",
                err=True,
            )

        # Discover direction from program.md
        direction = "maximize"
        program_md_path = hyp_dir / "program.md"
        if program_md_path.exists():
            try:
                from research_to_dev.experiment.program_reader import parse_program_md

                spec = parse_program_md(program_md_path)
                direction = spec.direction
            except (ValueError, FileNotFoundError):
                pass

        summary = compile_hypothesis(
            config.hypothesis_filter, iterations, direction
        )
        from datetime import datetime, timezone

        compiled = CompiledReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            hypotheses=[summary],
        )
    else:
        # All hypotheses mode
        compiled = compile_all(
            reader,
            experiments_dir,
            insights_generator=insights_generator,
            traceability=traceability_ctx,
        )

        if not compiled.hypotheses:
            typer.echo(
                "Error: No hypothesis results found. "
                "No results.tsv files discovered in any experiment subdirectory.",
                err=True,
            )
            raise typer.Exit(code=1)

    # -- 4. Insights failure check (when generation was attempted) --------
    if insights_generator is not None and compiled.insights is None:
        typer.echo(
            "Warning: Insights generation failed. "
            "Report compiled without insights.",
            err=True,
        )

    # -- 5. Write reports -------------------------------------------------
    md_path, json_path = write_reports(compiled, reports_dir)

    typer.echo(f"Report written to {md_path}")
    typer.echo(f"Report written to {json_path}")

    # -- 6. Terminal executive summary (TR-19) ----------------------------
    elapsed = time.monotonic() - t_start
    _print_report_summary(compiled, md_path, json_path, elapsed)


# ======================================================================
# Terminal summary rendering
# ======================================================================


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis if longer than max_len."""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _print_run_summary(
    results_path: Path, program_spec: object
) -> None:
    """Print a completion summary after the experiment loop finishes (AE-26).

    Reads the results TSV to compute iterations run, keeps/discards, and
    the final baseline value.  If no results were written (e.g. zero
    iterations) a "No iterations" message is shown instead.
    """
    from research_to_dev.experiment.program_reader import ProgramSpec

    assert isinstance(program_spec, ProgramSpec)

    direction = program_spec.direction
    metric_name = program_spec.target_metric

    # ----- read TSV rows ------------------------------------------------
    if not results_path.exists():
        typer.echo("\nExperiment completed. No iterations ran (results file not found).")
        return

    raw = results_path.read_text(encoding="utf-8")
    lines = raw.strip().split("\n")

    if len(lines) <= 1:
        # Header only, no data rows
        typer.echo("\nExperiment completed. No iterations ran.")
        return

    # Skip header (line 0)
    rows: list[dict] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 6:
            rows.append({
                "iteration": int(parts[0]),
                "metric_name": parts[1],
                "metric_value": float(parts[2]),
                "baseline_value": float(parts[3]),
                "delta": float(parts[4]),
                "status": parts[5],
            })

    if not rows:
        typer.echo("\nExperiment completed. No data rows in results file.")
        return

    # ----- compute keeps / discards -------------------------------------
    kept = 0
    discarded = 0
    failed = 0
    timeout_count = 0

    for row in rows:
        status = row["status"]
        if status == "timeout":
            timeout_count += 1
            discarded += 1  # timeouts are treated as discards
        elif status == "failed":
            failed += 1
            discarded += 1  # failed iterations are also discards
        else:
            # status == "success": use delta to infer keep/discard
            delta = row["delta"]
            if direction == "maximize":
                if delta >= 0:
                    kept += 1
                else:
                    discarded += 1
            else:  # minimize
                if delta <= 0:
                    kept += 1
                else:
                    discarded += 1

    # ----- final baseline -----------------------------------------------
    # The baseline in the TSV row is the *pre-iteration* baseline.
    # If the last iteration was a keep, the new baseline is the row's
    # metric_value; otherwise it's the row's baseline_value.
    last = rows[-1]
    last_delta = last["delta"]
    if direction == "maximize" and last_delta >= 0:
        final_baseline = last["metric_value"]
    elif direction == "minimize" and last_delta <= 0:
        final_baseline = last["metric_value"]
    else:
        final_baseline = last["baseline_value"]

    # ----- print --------------------------------------------------------
    typer.echo()
    typer.echo("═══════════════════════════════════════")
    typer.echo(" EXPERIMENT COMPLETED")
    typer.echo("═══════════════════════════════════════")
    typer.echo(f"  Iterations:     {len(rows)}")
    typer.echo(f"  Kept:           {kept} (improvements committed)")
    typer.echo(f"  Discarded:      {discarded} (failed/timeout/worse)")
    if failed:
        typer.echo(f"    └─ failed:    {failed}")
    if timeout_count:
        typer.echo(f"    └─ timeout:   {timeout_count}")
    typer.echo(f"  Final baseline: {metric_name}={final_baseline}")
    typer.echo("═══════════════════════════════════════")


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


# ======================================================================
# Report command helpers
# ======================================================================


def _compute_verdict(hypotheses: list) -> str:
    """Count improved vs declined vs neutral hypotheses.

    Returns a verdict string like ``"3/5 hypotheses improved vs baseline"``.

    Rules (rule-based, no direction needed — status already encodes it):
    - improved: HypothesisSummary.status == "improved"
    - declined: HypothesisSummary.status == "worsened"
    - neutral: status is "crash" or "no_iterations"
    """
    improved = 0
    total = len(hypotheses)
    for h in hypotheses:
        if h.status == "improved":
            improved += 1
    return f"{improved}/{total} hypotheses improved vs baseline"


def _print_report_summary(
    compiled,  # CompiledReport
    md_path: Path,
    json_path: Path,
    elapsed: float,
) -> None:
    """Print a human-readable executive summary after report compilation.

    Shows: top 5 hypotheses (title truncated, delta, status),
    top 3 recommendations (if insights available), overall verdict,
    output file paths, and elapsed time.

    Follows the same box-drawing pattern as ``_print_summary()`` for
    the ``analyze`` command. Does NOT include traceability data.
    """
    typer.echo()
    typer.echo("═══════════════════════════════════════")
    typer.echo(" REPORT SUMMARY")
    typer.echo("═══════════════════════════════════════")

    # -- Top hypotheses ---------------------------------------------------
    typer.echo()
    typer.echo(" Top Hypotheses:")
    typer.echo(f" {'#':>2}  {'Delta':>8}   {'Title':<50} {'Status':>8}")

    for i, h in enumerate(compiled.hypotheses[:5], 1):
        # Delta: format as signed percentage or "N/A"
        bc = h.baseline_comparison
        if bc is not None and bc.best_improvement is not None:
            delta_str = f"{bc.best_improvement:+.1%}"
        else:
            delta_str = "N/A"

        # Title: use hypothesis_id (same convention as markdown formatter)
        title_trunc = _truncate(h.hypothesis_id, 50)

        # Status
        status_str = h.status

        typer.echo(
            f" {i:>2}  {delta_str:>8}   {title_trunc:<50} {status_str:>8}"
        )

    # -- Top recommendations (if insights) --------------------------------
    if compiled.insights is not None and compiled.insights.recommendations:
        typer.echo()
        typer.echo(" Top Recommendations:")
        for j, rec in enumerate(compiled.insights.recommendations[:3], 1):
            action_trunc = _truncate(rec.action, 55)
            typer.echo(f"  {j}. {action_trunc}")

    # -- Verdict, output, time -------------------------------------------
    verdict = _compute_verdict(compiled.hypotheses)

    typer.echo()
    typer.echo(f" Verdict: {verdict}")
    typer.echo(f" Output:   {md_path}")
    typer.echo(f" Output:   {json_path}")
    typer.echo(f" Time:     {elapsed:.1f}s")
    typer.echo("═══════════════════════════════════════")
