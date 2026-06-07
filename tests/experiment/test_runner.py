"""Tests for ExperimentRunner — iteration loop, retry logic, timeout
handling, keep/discard decisions, and termination conditions (AE-19–AE-23).

Uses ``MagicMock(spec=...)`` for all dependencies so the runner can be
exercised without real subprocesses, git operations, or filesystem I/O.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from research_to_dev.experiment.agent import AgentResult
from research_to_dev.experiment.executor import ExecutionResult
from research_to_dev.experiment.metrics import decide_keep
from research_to_dev.experiment.program_reader import ProgramSpec
from research_to_dev.experiment.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def program_spec() -> ProgramSpec:
    """Minimal valid ProgramSpec for testing."""
    return ProgramSpec(
        hypothesis_id="test-hypothesis",
        target_metric="accuracy",
        success_criteria=">= 0.85",
        time_budget="30m",
        max_iterations=5,
        baseline={"accuracy": 0.72},
        run_command="python train.py",
        coding_agent_model="opencode-go/deepseek-v4-pro",
        time_budget_seconds=1800,
        direction="maximize",
        body="## Test body",
    )


@pytest.fixture
def mock_agent() -> MagicMock:
    """Mock CodingAgent — succeeds by default."""
    agent = MagicMock()
    agent.run.return_value = AgentResult(
        exit_code=0, stdout="ok", stderr="", files_changed=[], timed_out=False
    )
    return agent


@pytest.fixture
def mock_executor() -> MagicMock:
    """Mock CodeExecutor — succeeds by default, output includes metric."""
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        exit_code=0, stdout="accuracy: 0.85\n", stderr="", timed_out=False
    )
    return executor


@pytest.fixture
def mock_git_ops() -> MagicMock:
    """Mock GitOperations — clean branch by default."""
    git_ops = MagicMock()
    git_ops.get_current_branch.return_value = "experiment/test-hypothesis"
    return git_ops


@pytest.fixture
def mock_metric_registry() -> MagicMock:
    """Mock MetricRegistry — extracts accuracy: 0.85 by default."""
    reg = MagicMock()
    reg.extract.return_value = 0.85
    return reg


@pytest.fixture
def mock_results_writer() -> MagicMock:
    """Mock ResultsWriter — no-op."""
    return MagicMock()


@pytest.fixture
def runner(
    mock_agent: MagicMock,
    mock_executor: MagicMock,
    mock_git_ops: MagicMock,
    mock_metric_registry: MagicMock,
    mock_results_writer: MagicMock,
    program_spec: ProgramSpec,
) -> ExperimentRunner:
    """Build an ExperimentRunner with mock deps."""
    return ExperimentRunner(
        agent=mock_agent,
        executor=mock_executor,
        git_ops=mock_git_ops,
        metric_registry=mock_metric_registry,
        results_writer=mock_results_writer,
        program_spec=program_spec,
    )


# ---------------------------------------------------------------------------
# AE-19: Constructor and basic structure
# ---------------------------------------------------------------------------


class TestRunnerConstruction:
    """AE-19: ExperimentRunner constructor-injected deps."""

    def test_constructor_stores_all_deps(
        self,
        mock_agent: MagicMock,
        mock_executor: MagicMock,
        mock_git_ops: MagicMock,
        mock_metric_registry: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        runner = ExperimentRunner(
            agent=mock_agent,
            executor=mock_executor,
            git_ops=mock_git_ops,
            metric_registry=mock_metric_registry,
            results_writer=mock_results_writer,
            program_spec=program_spec,
        )
        assert runner._agent is mock_agent  # type: ignore[attr-defined]
        assert runner._executor is mock_executor  # type: ignore[attr-defined]
        assert runner._git_ops is mock_git_ops  # type: ignore[attr-defined]
        assert runner._metric_registry is mock_metric_registry  # type: ignore[attr-defined]
        assert runner._results is mock_results_writer  # type: ignore[attr-defined]
        assert runner._spec is program_spec  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# AE-20: Branch verification and iteration guards
# ---------------------------------------------------------------------------


class TestBranchVerification:
    """AE-20: runner verifies it is on the correct experiment branch."""

    def test_accepts_correct_branch(self, runner: ExperimentRunner, mock_git_ops: MagicMock) -> None:
        mock_git_ops.get_current_branch.return_value = "experiment/test-hypothesis"
        runner._verify_branch()  # should not raise

    def test_rejects_wrong_branch(self, runner: ExperimentRunner, mock_git_ops: MagicMock) -> None:
        mock_git_ops.get_current_branch.return_value = "main"
        with pytest.raises(RuntimeError, match="Expected to be on branch"):
            runner._verify_branch()

    def test_rejects_substring_mismatch(self, runner: ExperimentRunner, mock_git_ops: MagicMock) -> None:
        mock_git_ops.get_current_branch.return_value = "experiment/other-hypothesis"
        with pytest.raises(RuntimeError, match="Expected to be on branch"):
            runner._verify_branch()

    def test_run_verifies_branch_on_start(
        self,
        runner: ExperimentRunner,
        mock_git_ops: MagicMock,
    ) -> None:
        mock_git_ops.get_current_branch.return_value = "wrong-branch"
        with pytest.raises(RuntimeError, match="Expected to be on branch"):
            runner.run(Path("program.md"))


class TestIterationGuards:
    """AE-20: time budget and max_iterations guards."""

    def test_stops_after_max_iterations(
        self,
        runner: ExperimentRunner,
        mock_agent: MagicMock,
        mock_executor: MagicMock,
        mock_metric_registry: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        program_spec.max_iterations = 3
        runner = ExperimentRunner(
            agent=mock_agent,
            executor=mock_executor,
            git_ops=MagicMock(),
            metric_registry=mock_metric_registry,
            results_writer=mock_results_writer,
            program_spec=program_spec,
        )
        runner._git_ops.get_current_branch.return_value = "experiment/test-hypothesis"  # type: ignore[attr-defined]

        runner.run(Path("program.md"))

        # 3 iterations → 3 agent calls, 3 executor calls
        assert mock_agent.run.call_count == 3
        assert mock_executor.execute.call_count == 3

    def test_stops_when_time_budget_exhausted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        runner: ExperimentRunner,
        mock_agent: MagicMock,
        mock_git_ops: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Simulate elapsed time exceeding budget after first iteration."""
        program_spec.time_budget_seconds = 60
        program_spec.max_iterations = 100

        calls = 0

        def fake_monotonic() -> float:
            nonlocal calls
            calls += 1
            # call 1 = start time (0)
            # call 2 = iteration 1 remaining check (0 → 60s left → proceeds)
            # call 3 = iteration 2 remaining check (1000 → expired → breaks)
            if calls == 1:
                return 0.0
            if calls == 2:
                return 0.0  # still within budget for first iteration
            return 1000.0  # well past deadline

        import research_to_dev.experiment.runner as runner_mod

        monkeypatch.setattr(runner_mod.time, "monotonic", fake_monotonic)

        mock_agent.run.return_value = AgentResult(
            exit_code=0, stdout="ok", stderr="", files_changed=[], timed_out=False
        )

        runner.run(Path("program.md"))

        # Should break before second iteration due to time budget
        assert mock_agent.run.call_count == 1

    def test_time_budget_fractional_second(
        self,
        monkeypatch: pytest.MonkeyPatch,
        runner: ExperimentRunner,
        program_spec: ProgramSpec,
    ) -> None:
        """Fractional remaining time is floored to int for timeout (D5)."""
        program_spec.time_budget_seconds = 10
        program_spec.max_iterations = 1

        calls = 0

        def fake_monotonic() -> float:
            nonlocal calls
            calls += 1
            # call 1 = start time (0) → deadline=10
            # call 2 = remaining check (5.3 elapsed) → remaining=4.7 → int=4
            if calls == 1:
                return 0.0
            return 5.3

        import research_to_dev.experiment.runner as runner_mod

        monkeypatch.setattr(runner_mod.time, "monotonic", fake_monotonic)

        runner.run(Path("program.md"))

        # Agent receives timeout=int(10 - 5.3) = int(4.7) = 4
        assert runner._agent.run.call_args.kwargs["timeout"] == 4  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# AE-21: Within-iteration retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """AE-21: executor non-zero exit → retry max 2 times with stderr."""

    def test_no_retry_on_success(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_agent: MagicMock,
    ) -> None:
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0, stdout="accuracy: 0.85", stderr="", timed_out=False
        )

        runner.run(Path("program.md"))

        # 5 iterations × 1 executor call each
        assert mock_executor.execute.call_count == 5
        # Agent called once per iteration
        assert mock_agent.run.call_count == 5

    def test_retry_on_non_zero_exit_single(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_agent: MagicMock,
        mock_metric_registry: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """First attempt fails → one retry succeeds."""
        program_spec.max_iterations = 1

        # First attempt: non-zero exit with stderr
        # Second attempt (retry): success
        mock_executor.execute.side_effect = [
            ExecutionResult(exit_code=1, stdout="", stderr="TypeError: foo", timed_out=False),
            ExecutionResult(exit_code=0, stdout="accuracy: 0.88", stderr="", timed_out=False),
        ]

        runner.run(Path("program.md"))

        # Two executor calls: first attempt + retry
        assert mock_executor.execute.call_count == 2
        # One initial agent call + one retry agent call with stderr context
        assert mock_agent.run.call_count == 2

    def test_retry_pass_stderr_to_agent(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_agent: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """AE-21: retry prompt includes stderr as stack trace context."""
        program_spec.max_iterations = 1

        mock_executor.execute.side_effect = [
            ExecutionResult(exit_code=1, stdout="", stderr="TypeError: bar", timed_out=False),
            ExecutionResult(exit_code=0, stdout="accuracy: 0.88", stderr="", timed_out=False),
        ]

        runner.run(Path("program.md"))

        retry_call = mock_agent.run.call_args_list[1]
        assert "STACK TRACE" in retry_call.kwargs["prompt"]
        assert "TypeError: bar" in retry_call.kwargs["prompt"]

    def test_crash_after_two_retries_exhausted(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_agent: MagicMock,
        mock_results_writer: MagicMock,
        mock_git_ops: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Both retries fail → 'crash' logged as 'failed', discard, continue."""
        program_spec.max_iterations = 1

        mock_executor.execute.side_effect = [
            ExecutionResult(exit_code=1, stdout="", stderr="err", timed_out=False),
            ExecutionResult(exit_code=1, stdout="", stderr="err2", timed_out=False),
            ExecutionResult(exit_code=1, stdout="", stderr="err3", timed_out=False),
        ]

        runner.run(Path("program.md"))

        # All 3 attempts exhausted
        assert mock_executor.execute.call_count == 3
        # Initial agent + 2 retry agent calls
        assert mock_agent.run.call_count == 3

        # Status "failed" (crash)
        append_call = mock_results_writer.append.call_args
        assert append_call.kwargs["status"] == "failed"

        # Changes discarded
        mock_git_ops.reset_hard.assert_called()

    def test_retry_counter_resets_on_success(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_agent: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Iteration with retry → next iteration starts fresh."""
        program_spec.max_iterations = 2

        # Iteration 1: fails once, succeeds on retry
        # Iteration 2: succeeds immediately
        mock_executor.execute.side_effect = [
            ExecutionResult(exit_code=1, stdout="", stderr="e1", timed_out=False),
            ExecutionResult(exit_code=0, stdout="accuracy: 0.80", stderr="", timed_out=False),
            ExecutionResult(exit_code=0, stdout="accuracy: 0.81", stderr="", timed_out=False),
        ]

        runner.run(Path("program.md"))

        # Iteration 1: 1 initial fail + 1 retry. Iteration 2: 1 success.
        assert mock_executor.execute.call_count == 3
        # Iteration 1: 1 initial agent + 1 retry agent. Iteration 2: 1 initial agent
        assert mock_agent.run.call_count == 3


# ---------------------------------------------------------------------------
# AE-22: Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """AE-22: timeout → log 'timeout', discard, continue."""

    def test_agent_timeout_logs_and_continues(
        self,
        runner: ExperimentRunner,
        mock_agent: MagicMock,
        mock_executor: MagicMock,
        mock_results_writer: MagicMock,
        mock_git_ops: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        program_spec.max_iterations = 2

        # Iteration 1: agent timeout
        # Iteration 2: normal success
        mock_agent.run.side_effect = [
            AgentResult(exit_code=-1, stdout="", stderr="", timed_out=True),
            AgentResult(exit_code=0, stdout="ok", stderr="", timed_out=False),
        ]

        runner.run(Path("program.md"))

        # Only iteration 2 calls executor
        assert mock_executor.execute.call_count == 1

        # Iteration 1 timeout row
        timeout_calls = [
            c for c in mock_results_writer.append.call_args_list
            if c.kwargs["status"] == "timeout"
        ]
        assert len(timeout_calls) == 1
        assert timeout_calls[0].kwargs["iteration"] == 1

        # Iteration 2 success row
        success_calls = [
            c for c in mock_results_writer.append.call_args_list
            if c.kwargs["status"] == "success"
        ]
        assert len(success_calls) == 1
        assert success_calls[0].kwargs["iteration"] == 2

    def test_executor_timeout_logs_and_continues(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_results_writer: MagicMock,
        mock_git_ops: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        program_spec.max_iterations = 2

        # Iteration 1: executor timeout
        # Iteration 2: normal success
        mock_executor.execute.side_effect = [
            ExecutionResult(exit_code=-1, stdout="", stderr="", timed_out=True),
            ExecutionResult(exit_code=0, stdout="accuracy: 0.90", stderr="", timed_out=False),
        ]

        runner.run(Path("program.md"))

        timeout_calls = [
            c for c in mock_results_writer.append.call_args_list
            if c.kwargs["status"] == "timeout"
        ]
        assert len(timeout_calls) == 1
        assert timeout_calls[0].kwargs["iteration"] == 1

        success_calls = [
            c for c in mock_results_writer.append.call_args_list
            if c.kwargs["status"] == "success"
        ]
        assert len(success_calls) == 1
        assert success_calls[0].kwargs["iteration"] == 2

    def test_executor_timeout_discards_changes(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_git_ops: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        program_spec.max_iterations = 1
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=-1, stdout="", stderr="", timed_out=True
        )

        runner.run(Path("program.md"))

        mock_git_ops.reset_hard.assert_called()

    def test_agent_timeout_during_retry(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_agent: MagicMock,
        mock_results_writer: MagicMock,
        mock_git_ops: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Agent times out during retry → logged as timeout, discard."""
        program_spec.max_iterations = 1

        # First executor: non-zero exit → trigger retry
        mock_executor.execute.side_effect = [
            ExecutionResult(exit_code=1, stdout="", stderr="err", timed_out=False),
        ]
        # Agent times out on retry invocation
        mock_agent.run.side_effect = [
            AgentResult(exit_code=0, stdout="ok", stderr="", timed_out=False),
            AgentResult(exit_code=-1, stdout="", stderr="", timed_out=True),
        ]

        runner.run(Path("program.md"))

        # Result row is timeout
        append_call = mock_results_writer.append.call_args
        assert append_call.kwargs["status"] == "timeout"

    def test_remaining_budget_used_as_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        runner: ExperimentRunner,
        mock_agent: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """AE-22: remaining time_budget used as timeout parameter."""
        program_spec.time_budget_seconds = 300
        program_spec.max_iterations = 1

        import research_to_dev.experiment.runner as runner_mod

        calls = 0

        def fake_monotonic() -> float:
            nonlocal calls
            calls += 1
            # call 1 = start time (0) → deadline=300
            # call 2 = remaining check (120 elapsed) → remaining=180
            if calls == 1:
                return 0.0
            return 120.0

        monkeypatch.setattr(runner_mod.time, "monotonic", fake_monotonic)

        runner.run(Path("program.md"))

        assert mock_agent.run.call_args.kwargs["timeout"] == 180


# ---------------------------------------------------------------------------
# AE-23: Post-execution logic
# ---------------------------------------------------------------------------


class TestPostExecutionKeep:
    """AE-23: successful execution → extract, decide, commit/reset."""

    def test_keep_commits_and_updates_baseline(
        self,
        runner: ExperimentRunner,
        mock_agent: MagicMock,
        mock_executor: MagicMock,
        mock_metric_registry: MagicMock,
        mock_git_ops: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Improvement → commit, baseline updated (D9)."""
        program_spec.max_iterations = 2
        program_spec.direction = "maximize"
        program_spec.baseline = {"accuracy": 0.72}

        # Both iterations produce improvements
        mock_metric_registry.extract.side_effect = [0.80, 0.85]
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0, stdout="accuracy: 0.80", stderr="", timed_out=False
        )

        runner.run(Path("program.md"))

        # Two commits (both kept)
        assert mock_git_ops.commit_changes.call_count == 2

        # No resets (nothing discarded)
        mock_git_ops.reset_hard.assert_not_called()

        # Check baseline evolution via appended rows
        append_calls = mock_results_writer.append.call_args_list

        # Iteration 1: baseline=0.72, value=0.80 → keep
        assert append_calls[0].kwargs["baseline"] == 0.72
        assert append_calls[0].kwargs["value"] == 0.80
        assert append_calls[0].kwargs["status"] == "success"

        # Iteration 2: baseline=0.80 (updated), value=0.85 → keep
        assert append_calls[1].kwargs["baseline"] == 0.80
        assert append_calls[1].kwargs["value"] == 0.85
        assert append_calls[1].kwargs["status"] == "success"

    def test_discard_resets_and_preserves_baseline(
        self,
        runner: ExperimentRunner,
        mock_executor: MagicMock,
        mock_metric_registry: MagicMock,
        mock_git_ops: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Worsening → reset_hard, baseline unchanged."""
        program_spec.max_iterations = 1
        program_spec.direction = "maximize"
        program_spec.baseline = {"accuracy": 0.72}

        mock_metric_registry.extract.return_value = 0.65  # worse than 0.72

        runner.run(Path("program.md"))

        # Discard → reset called
        mock_git_ops.reset_hard.assert_called_once()
        # No commit
        mock_git_ops.commit_changes.assert_not_called()

        # Baseline stays at 0.72
        append_call = mock_results_writer.append.call_args
        assert append_call.kwargs["baseline"] == 0.72
        assert append_call.kwargs["value"] == 0.65
        assert append_call.kwargs["status"] == "success"

    def test_delta_calculation_positive(
        self,
        runner: ExperimentRunner,
        mock_metric_registry: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Delta = current_value - baseline_value."""
        program_spec.max_iterations = 1
        program_spec.baseline = {"accuracy": 0.72}
        mock_metric_registry.extract.return_value = 0.85

        runner.run(Path("program.md"))

        assert mock_results_writer.append.call_args.kwargs["delta"] == pytest.approx(0.13)

    def test_delta_calculation_negative(
        self,
        runner: ExperimentRunner,
        mock_metric_registry: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Delta can be negative when metric worsened."""
        program_spec.max_iterations = 1
        program_spec.baseline = {"accuracy": 0.72}
        mock_metric_registry.extract.return_value = 0.60

        runner.run(Path("program.md"))

        assert mock_results_writer.append.call_args.kwargs["delta"] == pytest.approx(-0.12)

    def test_keep_then_discard_shape(
        self,
        runner: ExperimentRunner,
        mock_agent: MagicMock,
        mock_executor: MagicMock,
        mock_metric_registry: MagicMock,
        mock_git_ops: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """AE-32 scenario: iteration 1 keeps, iteration 2 discards."""
        program_spec.max_iterations = 2
        program_spec.direction = "maximize"
        program_spec.baseline = {"accuracy": 0.72}

        mock_metric_registry.extract.side_effect = [0.80, 0.70]
        mock_executor.execute.return_value = ExecutionResult(
            exit_code=0, stdout="ok", stderr="", timed_out=False
        )

        runner.run(Path("program.md"))

        # Iteration 1: commit (keep)
        # Iteration 2: reset (discard)
        assert mock_git_ops.commit_changes.call_count == 1
        assert mock_git_ops.reset_hard.call_count == 1


class TestPostExecutionMetricFailure:
    """AE-23: metric extraction failure → 'failed', discard."""

    def test_metric_extraction_fails_logs_failed(
        self,
        runner: ExperimentRunner,
        mock_metric_registry: MagicMock,
        mock_executor: MagicMock,
        mock_git_ops: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        program_spec.max_iterations = 1
        mock_metric_registry.extract.return_value = None  # no match

        runner.run(Path("program.md"))

        assert mock_results_writer.append.call_args.kwargs["status"] == "failed"
        mock_git_ops.reset_hard.assert_called_once()
        mock_git_ops.commit_changes.assert_not_called()


class TestPostExecutionNotification:
    """D9: user is notified on keep via print."""

    def test_keep_prints_notification(
        self,
        runner: ExperimentRunner,
        mock_metric_registry: MagicMock,
        program_spec: ProgramSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        program_spec.max_iterations = 1
        program_spec.baseline = {"accuracy": 0.72}
        mock_metric_registry.extract.return_value = 0.85

        runner.run(Path("program.md"))

        captured = capsys.readouterr()
        assert "KEEP" in captured.out
        assert "accuracy" in captured.out
        assert "0.72" in captured.out
        assert "0.85" in captured.out

    def test_discard_does_not_print_notification(
        self,
        runner: ExperimentRunner,
        mock_metric_registry: MagicMock,
        program_spec: ProgramSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        program_spec.max_iterations = 1
        program_spec.direction = "maximize"
        program_spec.baseline = {"accuracy": 0.72}
        mock_metric_registry.extract.return_value = 0.50  # worse

        runner.run(Path("program.md"))

        captured = capsys.readouterr()
        assert "KEEP" not in captured.out


# ---------------------------------------------------------------------------
# AE-23 + D9: Baseline mismatch
# ---------------------------------------------------------------------------


class TestBaselineMismatch:
    """Runner raises clean error when target_metric not in baseline."""

    def test_target_not_in_baseline(
        self,
        mock_agent: MagicMock,
        mock_executor: MagicMock,
        mock_git_ops: MagicMock,
        mock_metric_registry: MagicMock,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        program_spec.target_metric = "val_loss"
        program_spec.baseline = {"accuracy": 0.72}  # mismatch

        runner = ExperimentRunner(
            agent=mock_agent,
            executor=mock_executor,
            git_ops=mock_git_ops,
            metric_registry=mock_metric_registry,
            results_writer=mock_results_writer,
            program_spec=program_spec,
        )

        with pytest.raises(ValueError, match="val_loss.*not present.*baseline"):
            runner.run(Path("program.md"))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge-case behavior."""

    def test_single_iteration(
        self,
        runner: ExperimentRunner,
        mock_results_writer: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        program_spec.max_iterations = 1
        runner.run(Path("program.md"))
        assert mock_results_writer.append.call_count == 1

    def test_zero_iterations_config_runs_nothing(
        self,
        mock_agent: MagicMock,
        mock_git_ops: MagicMock,
        mock_metric_registry: MagicMock,
        mock_executor: MagicMock,
        mock_results_writer: MagicMock,
    ) -> None:
        """max_iterations=0 is rejected by ProgramReader, but if somehow
        set to 0, range(1, 1) produces no iterations."""
        mock_git_ops.get_current_branch.return_value = "experiment/test"
        spec = ProgramSpec(
            hypothesis_id="test",
            target_metric="accuracy",
            success_criteria=">= 0.5",
            time_budget="10m",
            max_iterations=0,
            baseline={"accuracy": 0.5},
            run_command="echo hi",
            coding_agent_model="model",
            time_budget_seconds=600,
            direction="maximize",
            body="body",
        )
        runner = ExperimentRunner(
            agent=mock_agent,
            executor=mock_executor,
            git_ops=mock_git_ops,
            metric_registry=mock_metric_registry,
            results_writer=mock_results_writer,
            program_spec=spec,
        )

        runner.run(Path("program.md"))

        mock_agent.run.assert_not_called()

    def test_minimize_direction_keep(
        self,
        runner: ExperimentRunner,
        mock_metric_registry: MagicMock,
        mock_git_ops: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Minimize direction: lower value is keep."""
        program_spec.max_iterations = 1
        program_spec.direction = "minimize"
        program_spec.baseline = {"accuracy": 0.50}
        mock_metric_registry.extract.return_value = 0.30  # lower = better

        runner.run(Path("program.md"))

        mock_git_ops.commit_changes.assert_called_once()

    def test_equal_values_are_keep(
        self,
        runner: ExperimentRunner,
        mock_metric_registry: MagicMock,
        mock_git_ops: MagicMock,
        program_spec: ProgramSpec,
    ) -> None:
        """Equal value → keep (don't discard progress that can't yet beat)."""
        program_spec.max_iterations = 1
        program_spec.baseline = {"accuracy": 0.80}
        mock_metric_registry.extract.return_value = 0.80

        runner.run(Path("program.md"))

        mock_git_ops.commit_changes.assert_called_once()
