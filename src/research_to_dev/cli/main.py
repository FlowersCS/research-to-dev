"""CLI entry points for the research-to-dev tool.

Uses ``typer`` — first CLI module in the project.
"""

from __future__ import annotations

import asyncio
import os

import typer
from openai import AsyncOpenAI

from research_to_dev.codebase.adapters import OpenAIDescriber
from research_to_dev.codebase.pipeline import CodebaseAnalysisPipeline
from research_to_dev.shared.config import CodebaseConfig

app = typer.Typer(
    name="research-to-dev",
    help="Research-to-development pipeline: analyze codebases and correlate with papers.",
)

codebase_app = typer.Typer(
    help="Codebase exploration and analysis commands.",
)
app.add_typer(codebase_app, name="codebase")


@codebase_app.command(name="analyze")
def analyze_codebase(
    path: str = typer.Argument(
        ".",
        help="Path to the project directory to analyze (default: current directory).",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="OpenAI model to use for component descriptions.",
    ),
    include_private: bool = typer.Option(
        False,
        "--include-private/--no-include-private",
        help="Include _-prefixed (private) functions and classes.",
    ),
    max_components: int = typer.Option(
        200,
        "--max-components",
        help="Maximum components to extract per module.",
    ),
) -> None:
    """Analyze a Python codebase and generate component descriptions.

    Scans the project directory for Python files, extracts functions,
    classes, and methods via AST parsing, and uses an LLM to generate
    natural language descriptions and module summaries.
    """
    resolved_path = os.path.abspath(path)
    if not os.path.isdir(resolved_path):
        typer.echo(f"Error: '{path}' is not a valid directory.", err=True)
        raise typer.Exit(code=1)

    config = CodebaseConfig(
        include_private=include_private,
        max_components_per_module=max_components,
        model=model,
    )

    client = AsyncOpenAI()
    describer = OpenAIDescriber(client=client, model=model)
    pipeline = CodebaseAnalysisPipeline(describer, config=config)

    result = asyncio.run(pipeline.analyze(resolved_path))

    # Print summary
    typer.echo(f"Project: {result.project_name}")
    typer.echo(f"Path: {result.project_path}")
    typer.echo(f"Language: {result.language}")
    typer.echo(f"Files scanned: {result.total_files_scanned}")
    typer.echo(f"Components found: {result.total_components_found}")
    typer.echo(f"Modules described: {len(result.modules)}")

    if result.modules:
        typer.echo("\nModule Summaries:")
        for mod in result.modules:
            summary = mod.summary[:100] + "..." if len(mod.summary) > 100 else mod.summary
            typer.echo(f"  [{mod.module_name}] {summary}")
