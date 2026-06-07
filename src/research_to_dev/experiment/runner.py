"""ExperimentRunner — orchestrates the agentic iteration loop (AE-19–AE-23).

Follows AD-02: single-responsibility orchestrator that coordinates the
coding agent, code executor, git operations, metric registry, and results
writer.  All dependencies are constructor-injected for full testability.
ProgramSpec (parsed program.md) is the single source of truth (D11).

Usage::

    from research_to_dev.experiment.runner import ExperimentRunner

    runner = ExperimentRunner(
        agent=adapter,
        executor=executor,
        git_ops=git_ops,
        metric_registry=registry,
        results_writer=writer,
        program_spec=spec,
    )
    runner.run(Path("program.md"))
"""

from __future__ import annotations

import time
from pathlib import Path

from research_to_dev.experiment.metrics import decide_keep
from research_to_dev.experiment.program_reader import ProgramSpec

# Import for type annotations — avoids circular imports at runtime.
# Sub-module imports for internal use are done via TYPE_CHECKING.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_to_dev.experiment.agent import CodingAgent
    from research_to_dev.experiment.executor import CodeExecutor
    from research_to_dev.experiment.git import GitOperations
    from research_to_dev.experiment.metrics import MetricRegistry
    from research_to_dev.experiment.results import ResultsWriter


class ExperimentRunner:
    """Orchestrates the experiment iteration loop (AD-02).

    Constructor-injected with all dependencies so every collaborator can
    be replaced with a test double.  Reads its configuration from the
    ``ProgramSpec`` (parsed program.md frontmatter per D11) — the loop
    itself is duration-agnostic (D5).

    The ``run()`` method drives the main loop::

        for each iteration (1 … max_iterations):
            └─ agent (prompt = abs path to program.md, D7)
            └─ executor (retry ≤2x on non-zero exit with stderr, D8)
            └─ extract metric
            └─ decide keep/discard (direction-aware)
            └─ commit (keep) or reset (discard)
            └─ append results row (D10)

    Error resilience follows T1–T4 as established in the design:
      * T1 (timeout):  log "timeout", discard, continue.
      * T2 (crash):    retry ≤2x within iteration; if exhausted → "failed".
      * T4 (git fail): RuntimeError propagates immediately (manual fix needed).
    """

    def __init__(
        self,
        *,
        agent: CodingAgent,
        executor: CodeExecutor,
        git_ops: GitOperations,
        metric_registry: MetricRegistry,
        results_writer: ResultsWriter,
        program_spec: ProgramSpec,
    ) -> None:
        self._agent = agent
        self._executor = executor
        self._git_ops = git_ops
        self._metric_registry = metric_registry
        self._results = results_writer
        self._spec = program_spec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, program_md_path: Path) -> None:
        """Run the full experiment iteration loop.

        Args:
            program_md_path: Absolute path to ``program.md``.  Used as the
                agent prompt (D7).  Must be an existing file.

        Raises:
            RuntimeError: If branch verification fails (not on the
                expected experiment branch), if the agent binary is
                unavailable, or if a git operation fails mid-loop (T4).
            ValueError: If ``target_metric`` is not present in the
                ``baseline`` dict from program.md frontmatter.
        """
        # AE-20: branch verification
        self._verify_branch()

        # Resolve configuration from ProgramSpec (single source, D11)
        target = self._spec.target_metric
        direction = self._spec.direction

        baseline_dict = self._spec.baseline
        try:
            baseline_value: float = baseline_dict[target]
        except KeyError:
            raise ValueError(
                f"target_metric '{target}' is not present in the baseline "
                f"dict {baseline_dict}.  Add a baseline entry for "
                f"'{target}' in program.md frontmatter."
            ) from None

        program_md_abs = str(program_md_path.resolve())
        working_dir: Path = program_md_path.parent

        # Time tracking (D5: loop sees integer seconds)
        start = time.monotonic()
        deadline = start + self._spec.time_budget_seconds

        # ------------------------------------------------------------------
        # Main iteration loop
        # ------------------------------------------------------------------
        for i in range(1, self._spec.max_iterations + 1):
            # AE-20: time budget guard
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            remaining_int = int(remaining)

            # --- Agent invocation ---
            agent_result = self._agent.run(
                prompt=program_md_abs,
                working_dir=working_dir,
                timeout=remaining_int,
                model=self._spec.coding_agent_model,
            )

            # AE-22: agent timeout → discard, continue
            if agent_result.timed_out:
                self._results.append(
                    iteration=i,
                    metric_name=target,
                    value=0.0,
                    baseline=baseline_value,
                    delta=0.0,
                    status="timeout",
                )
                self._discard_changes()
                continue

            # --- Executor invocation with within-iteration retry (D8) ---
            exec_outcome = self._run_executor_with_retry(
                iteration=i,
                remaining=remaining,
                program_md_abs=program_md_abs,
                working_dir=working_dir,
                target=target,
                baseline_value=baseline_value,
            )

            # Timeout / crash handled inside _run_executor_with_retry
            # (returns None for those cases).
            if exec_outcome is None:
                continue

            current_value = exec_outcome

            # AE-23: decide keep/discard
            keep = decide_keep(current_value, baseline_value, direction)
            delta = current_value - baseline_value

            if keep:
                # AE-23 step 3: commit, update baseline, notify (D9)
                self._git_ops.commit_changes(
                    f"iteration {i}: {target}={current_value}"
                )
                old_baseline = baseline_value
                baseline_value = current_value
                print(
                    f"Iteration {i}: KEEP — {target} improved "
                    f"from {old_baseline} to {current_value} "
                    f"(delta={delta:+.6f})"
                )

                self._results.append(
                    iteration=i,
                    metric_name=target,
                    value=current_value,
                    baseline=old_baseline,
                    delta=delta,
                    status="success",
                )
            else:
                # AE-23 step 4: discard (reset working tree)
                self._git_ops.reset_hard()

                self._results.append(
                    iteration=i,
                    metric_name=target,
                    value=current_value,
                    baseline=baseline_value,
                    delta=delta,
                    status="success",
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_branch(self) -> None:
        """AE-20: Ensure we are on the correct experiment branch."""
        current = self._git_ops.get_current_branch()
        expected = f"experiment/{self._spec.hypothesis_id}"
        if current != expected:
            raise RuntimeError(
                f"Expected to be on branch '{expected}' but currently on "
                f"'{current}'.  Run 'research-to-dev experiment setup' "
                f"first, or switch to the correct branch."
            )

    def _run_executor_with_retry(
        self,
        *,
        iteration: int,
        remaining: float,
        program_md_abs: str,
        working_dir: Path,
        target: str,
        baseline_value: float,
    ) -> float | None:
        """Run the executor with up to 2 within-iteration retries (AE-21, D8).

        On non-zero exit the agent is re-invoked with stderr context and
        the executor is retried.  Hard timeouts are caught immediately
        (T1) — no retries for timeout.

        Returns:
            The extracted metric value on success, or ``None`` if the
            iteration should be discarded (timeout, crash, or extraction
            failure).  In the ``None`` case the results row has already
            been appended.
        """
        remaining_int = int(remaining)
        run_cmd = self._spec.run_command

        exec_stdout = ""
        exec_stderr = ""
        exec_timed_out = False
        exec_success = False

        for retry in range(3):  # 0 = first attempt, 1–2 = retries
            exec_result = self._executor.execute(
                run_command=run_cmd,
                timeout=remaining_int,
            )

            # AE-22: executor timeout
            if exec_result.timed_out:
                exec_timed_out = True
                break

            # Exit 0 → success
            if exec_result.exit_code == 0:
                exec_success = True
                exec_stdout = exec_result.stdout
                exec_stderr = exec_result.stderr
                break

            # Non-zero exit — retry if we have retries left (D8)
            if retry < 2:
                retry_prompt = (
                    f"{program_md_abs}\n\n"
                    f"STACK TRACE from failed execution "
                    f"(attempt {retry + 1}):\n"
                    f"{exec_result.stderr}"
                )
                agent_result = self._agent.run(
                    prompt=retry_prompt,
                    working_dir=working_dir,
                    timeout=remaining_int,
                    model=self._spec.coding_agent_model,
                )

                if agent_result.timed_out:
                    exec_timed_out = True
                    break
            else:
                # Last retry — capture stderr for logging
                exec_stderr = exec_result.stderr

        # --- Handle timeout ---
        if exec_timed_out:
            self._results.append(
                iteration=iteration,
                metric_name=target,
                value=0.0,
                baseline=baseline_value,
                delta=0.0,
                status="timeout",
            )
            self._discard_changes()
            return None

        # --- Handle crash (non-zero exit after all retries) ---
        if not exec_success:
            self._results.append(
                iteration=iteration,
                metric_name=target,
                value=0.0,
                baseline=baseline_value,
                delta=0.0,
                status="failed",
            )
            self._discard_changes()
            return None

        # --- Extract metric ---
        current_value = self._metric_registry.extract(exec_stdout, target)

        if current_value is None:
            self._results.append(
                iteration=iteration,
                metric_name=target,
                value=0.0,
                baseline=baseline_value,
                delta=0.0,
                status="failed",
            )
            self._discard_changes()
            return None

        return current_value

    def _discard_changes(self) -> None:
        """Discard working-tree changes via git checkout (GE-02)."""
        self._git_ops.reset_hard()
