"""Program.md generation — structural transformation (0 LLM per D4).

Generates a YAML frontmatter + Markdown body file from a ``Hypothesis``
dataclass and ``ExperimentConfig``.  The success criteria appear in
both the frontmatter (canonical, machine-readable) and the objective
prose (human-readable) per D9.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from research_to_dev.shared.config import ExperimentConfig
from research_to_dev.hypothesis.types import Hypothesis


class ProgramWriter:
    """Generates and writes ``program.md`` for an experiment.

    Constructor-injectable so tests can supply arbitrary output
    directories without touching the real filesystem.

    Usage::

        writer = ProgramWriter(base_dir=".research-to-dev/experiments")
        path = writer.write(hypothesis, config, "accuracy")
    """

    def __init__(self, base_dir: str = ".research-to-dev/experiments") -> None:
        self._base = Path(base_dir)

    def write(
        self,
        hypothesis: Hypothesis,
        config: ExperimentConfig,
        target_metric: str,
        *,
        run_command: str,
        baseline: dict[str, float],
        coding_agent_model: str,
        direction: str | None = None,
    ) -> Path:
        """Generate and write ``program.md`` for a hypothesis.

        Args:
            hypothesis: The hypothesis to create the experiment for.
            config: Experiment configuration (time budget, max iterations).
            target_metric: The validated target metric name.
            run_command: Shell command to execute (D3).  Must not be empty.
            baseline: Baseline metric values as a dict (D4).
                Must not be empty.
            coding_agent_model: Model identifier for the coding agent (D12).
                Must not be empty.
            direction: Optional explicit direction override (D6).
                If ``None``, omitted from frontmatter — ``program_reader``
                will infer it from ``success_criteria``.

        Returns:
            Path to the created ``program.md`` file.

        Raises:
            ValueError: If any required parameter is empty or invalid.
        """
        # Fail loud for mandatory fields (T3)
        if not run_command.strip():
            raise ValueError("run_command must not be empty (D3).")
        if not baseline:
            raise ValueError("baseline must not be empty (D4).")
        if not coding_agent_model.strip():
            raise ValueError("coding_agent_model must not be empty (D12).")

        output_dir = self._base / hypothesis.id
        output_dir.mkdir(parents=True, exist_ok=True)

        content = self._render(
            hypothesis, config, target_metric,
            run_command=run_command, baseline=baseline,
            coding_agent_model=coding_agent_model, direction=direction,
        )
        file_path = output_dir / "program.md"
        file_path.write_text(content, encoding="utf-8")
        return file_path

    # ------------------------------------------------------------------
    # Private rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        hypothesis: Hypothesis,
        config: ExperimentConfig,
        target_metric: str,
        *,
        run_command: str,
        baseline: dict[str, float],
        coding_agent_model: str,
        direction: str | None = None,
    ) -> str:
        """Render the full program.md content.

        Returns a string with YAML frontmatter followed by Markdown body.
        """
        # --- YAML frontmatter ---
        frontmatter = {
            "hypothesis_id": hypothesis.id,
            "target_metric": target_metric,
            "success_criteria": hypothesis.success_criteria,
            "time_budget": config.time_budget,
            "max_iterations": config.max_iterations,
            "run_command": run_command,
            "baseline": baseline,
            "coding_agent_model": coding_agent_model,
        }
        # direction is optional — omitted if not explicitly provided (D6)
        # program_reader will infer it from success_criteria operators
        if direction is not None:
            frontmatter["direction"] = direction

        # --- Objective prose ---
        papers = hypothesis.supporting_papers or []
        papers_text = "\n".join(f"- {p}" for p in papers) if papers else "- (none)"

        objective = _render_objective(hypothesis, papers_text)

        # --- Agent Instructions (fixed template per D7) ---
        instructions = _render_agent_instructions(config, target_metric)

        # --- Assemble ---
        yaml_block = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip()

        return (
            f"---\n"
            f"{yaml_block}\n"
            f"---\n\n"
            f"{objective}\n\n"
            f"{instructions}\n"
        )


# ------------------------------------------------------------------
# Template helpers (module-level functions for testability)
# ------------------------------------------------------------------


def _render_objective(hypothesis: Hypothesis, papers_text: str) -> str:
    """Render the Experiment Objective section (structural transformation, D4)."""
    return (
        f"## Experiment Objective\n\n"
        f"**Title**: {hypothesis.title}\n\n"
        f"**Description**: {hypothesis.description}\n\n"
        f"**Approach**: {hypothesis.approach or '(not specified)'}\n\n"
        f"**Expected Improvement**: {hypothesis.expected_improvement or '(not specified)'}\n\n"
        f"**Code Changes**: {hypothesis.code_changes or '(not specified)'}\n\n"
        f"**Supporting Papers**:\n{papers_text}\n\n"
        f"**Success Criteria**: {hypothesis.success_criteria or '(not specified)'}"
    )


def _render_agent_instructions(config: ExperimentConfig, target_metric: str) -> str:
    """Render the fixed Agent Instructions template (D7)."""
    return (
        f"## Agent Instructions\n\n"
        f"1. Read the objective above. You are testing a hypothesis about code improvement.\n"
        f"2. Modify the codebase to implement the approach described.\n"
        f"3. Run the training/evaluation command. The target metric is `{target_metric}`.\n"
        f"4. Report the metric value after each run.\n"
        f"5. Time budget: {config.time_budget}. Max iterations: {config.max_iterations}.\n"
        f"6. Stop when success criteria are met or constraints exhausted.\n"
    )
