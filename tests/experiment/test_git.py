"""Tests for GitOperations — dirty check, branch creation, commit,
reset, and branch queries.

Uses real git repos in temp directories for integration-level testing,
plus subprocess mocking for unit-level error-path coverage.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_to_dev.experiment.git import GitOperations


@pytest.fixture
def git_repo() -> str:
    """Create a temporary git repo and return its path."""
    tmpdir = tempfile.mkdtemp()
    subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True)
    # Configure git user for commit (required by some git versions)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmpdir, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tester"],
        cwd=tmpdir, check=True, capture_output=True,
    )
    # Create an initial commit so we have a clean working tree
    (Path(tmpdir) / "README.md").write_text("# Test Repo")
    subprocess.run(["git", "add", "."], cwd=tmpdir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmpdir, check=True, capture_output=True,
    )
    yield tmpdir
    # cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestIsDirty:
    """is_dirty() detects clean vs dirty working tree."""

    def test_clean_repo_is_not_dirty(self, git_repo: str) -> None:
        """Fresh repo with no changes is not dirty."""
        ops = GitOperations(repo_path=git_repo)
        assert ops.is_dirty() is False

    def test_uncommitted_change_is_dirty(self, git_repo: str) -> None:
        """Adding a new untracked file makes the repo dirty."""
        ops = GitOperations(repo_path=git_repo)
        (Path(git_repo) / "new_file.txt").write_text("hello")
        assert ops.is_dirty() is True

    def test_modified_tracked_file_is_dirty(self, git_repo: str) -> None:
        """Modifying an existing tracked file makes the repo dirty."""
        ops = GitOperations(repo_path=git_repo)
        readme = Path(git_repo) / "README.md"
        readme.write_text("# Modified")
        assert ops.is_dirty() is True


class TestIsRepo:
    """is_repo() detects whether inside a git repo."""

    def test_inside_git_repo_returns_true(self, git_repo: str) -> None:
        """is_repo returns True inside a git repo."""
        ops = GitOperations(repo_path=git_repo)
        assert ops.is_repo() is True

    def test_outside_git_repo_returns_false(self) -> None:
        """is_repo returns False outside a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ops = GitOperations(repo_path=tmpdir)
            assert ops.is_repo() is False


class TestBranchExists:
    """branch_exists() checks for local branch existence."""

    def test_branch_does_not_exist(self, git_repo: str) -> None:
        """A non-existent branch returns False."""
        ops = GitOperations(repo_path=git_repo)
        assert ops.branch_exists("experiment/does-not-exist") is False

    def test_branch_exists(self, git_repo: str) -> None:
        """An existing branch returns True."""
        subprocess.run(
            ["git", "branch", "existing-branch"],
            cwd=git_repo, check=True, capture_output=True,
        )
        ops = GitOperations(repo_path=git_repo)
        assert ops.branch_exists("existing-branch") is True


class TestCreateBranch:
    """create_branch() creates a new git branch and switches to it."""

    def test_create_branch_succeeds_on_clean_repo(self, git_repo: str) -> None:
        """On a clean repo, create_branch creates the branch and switches."""
        ops = GitOperations(repo_path=git_repo)
        ops.create_branch("experiment/test-branch")

        # Verify we're on the new branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=git_repo, capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "experiment/test-branch"

    def test_create_branch_raises_on_dirty_repo(self, git_repo: str) -> None:
        """On a dirty repo, create_branch raises RuntimeError."""
        ops = GitOperations(repo_path=git_repo)
        (Path(git_repo) / "dirty_file.txt").write_text("uncommitted")

        with pytest.raises(RuntimeError, match="uncommitted changes"):
            ops.create_branch("experiment/test-branch")

    def test_create_branch_raises_on_existing_branch(self, git_repo: str) -> None:
        """If branch already exists, create_branch raises RuntimeError."""
        # Pre-create the branch
        subprocess.run(
            ["git", "branch", "experiment/already-exists"],
            cwd=git_repo, check=True, capture_output=True,
        )

        ops = GitOperations(repo_path=git_repo)
        with pytest.raises(RuntimeError, match="already exists"):
            ops.create_branch("experiment/already-exists")

    def test_create_branch_with_slash_in_name(self, git_repo: str) -> None:
        """Branch names with slashes are supported."""
        ops = GitOperations(repo_path=git_repo)
        ops.create_branch("experiment/abc123/feature")

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=git_repo, capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "experiment/abc123/feature"


# ---------------------------------------------------------------------------
# Phase 3: Git extensions (AE-13, AE-14, AE-15)
# ---------------------------------------------------------------------------


class TestCommitChanges:
    """commit_changes() stages all changes and commits with message."""

    def test_commits_with_given_message(self, git_repo: str) -> None:
        """Changes are staged and committed with the provided message."""
        ops = GitOperations(repo_path=git_repo)
        (Path(git_repo) / "new_file.txt").write_text("hello")
        ops.commit_changes("Iteration 1 keep")

        # Verify commit exists with the right message
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=git_repo, capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "Iteration 1 keep"

    def test_no_changes_commits_cleanly(self, git_repo: str) -> None:
        """Committing with no changes should raise RuntimeError (empty commit)."""
        ops = GitOperations(repo_path=git_repo)
        with pytest.raises(RuntimeError, match="Failed to commit"):
            ops.commit_changes("Empty commit")

    @patch("subprocess.run")
    def test_stages_all_changes(self, mock_run: MagicMock) -> None:
        """git add -A is invoked before git commit."""
        ok_result = MagicMock()
        ok_result.returncode = 0
        ok_result.stderr = ""
        mock_run.return_value = ok_result

        ops = GitOperations(repo_path="/fake/repo")
        ops.commit_changes("message")

        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0].args[0] == ["git", "add", "-A"]
        assert mock_run.call_args_list[1].args[0] == ["git", "commit", "-m", "message"]

    @patch("subprocess.run")
    def test_raises_runtime_error_on_add_failure(self, mock_run: MagicMock) -> None:
        """RuntimeError raised when git add fails (AE-15, T4)."""
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "fatal: not a git repository"
        mock_run.return_value = fail_result

        ops = GitOperations(repo_path="/fake/repo")
        with pytest.raises(RuntimeError, match="Failed to stage changes"):
            ops.commit_changes("message")

    @patch("subprocess.run")
    def test_raises_runtime_error_on_commit_failure(self, mock_run: MagicMock) -> None:
        """RuntimeError raised when git commit fails (AE-15, T4)."""
        ok_result = MagicMock()
        ok_result.returncode = 0
        ok_result.stderr = ""

        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "fatal: unable to commit"

        mock_run.side_effect = [ok_result, fail_result]

        ops = GitOperations(repo_path="/fake/repo")
        with pytest.raises(RuntimeError, match="Failed to commit changes"):
            ops.commit_changes("message")


class TestResetHard:
    """reset_hard() discards unstaged working tree changes via git checkout ."""

    def test_discards_unstaged_changes(self, git_repo: str) -> None:
        """Modifications to tracked files are reverted."""
        ops = GitOperations(repo_path=git_repo)
        readme = Path(git_repo) / "README.md"
        original = readme.read_text()
        readme.write_text("# Modified by agent")

        ops.reset_hard()

        assert readme.read_text() == original
        assert ops.is_dirty() is False

    @patch("subprocess.run")
    def test_runs_git_checkout_dot(self, mock_run: MagicMock) -> None:
        """reset_hard invokes git checkout . correctly."""
        ok_result = MagicMock()
        ok_result.returncode = 0
        ok_result.stderr = ""
        mock_run.return_value = ok_result

        ops = GitOperations(repo_path="/fake/repo")
        ops.reset_hard()

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["git", "checkout", "."]

    @patch("subprocess.run")
    def test_raises_runtime_error_on_failure(self, mock_run: MagicMock) -> None:
        """RuntimeError raised when git checkout . fails (AE-15, T4)."""
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "fatal: not a git repository"
        mock_run.return_value = fail_result

        ops = GitOperations(repo_path="/fake/repo")
        with pytest.raises(RuntimeError, match="Failed to discard working tree"):
            ops.reset_hard()


class TestCheckout:
    """checkout() is an alias for reset_hard()."""

    def test_is_alias_for_reset_hard(self, git_repo: str) -> None:
        """checkout() calls reset_hard() internally."""
        ops = GitOperations(repo_path=git_repo)
        readme = Path(git_repo) / "README.md"
        original = readme.read_text()
        readme.write_text("# Modified by agent")

        ops.checkout()  # same as reset_hard

        assert readme.read_text() == original

    @patch("subprocess.run")
    def test_raises_runtime_error_on_failure(self, mock_run: MagicMock) -> None:
        """RuntimeError propagates through the alias (AE-15, T4)."""
        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "fatal: not a git repository"
        mock_run.return_value = fail_result

        ops = GitOperations(repo_path="/fake/repo")
        with pytest.raises(RuntimeError, match="Failed to discard working tree"):
            ops.checkout()


class TestGetCurrentBranch:
    """get_current_branch() returns the active branch name."""

    def test_returns_main_branch(self, git_repo: str) -> None:
        """Returns the default branch name."""
        ops = GitOperations(repo_path=git_repo)
        branch = ops.get_current_branch()

        assert branch in ("main", "master")

    def test_returns_custom_branch_after_switch(self, git_repo: str) -> None:
        """Returns the branch name after switching."""
        ops = GitOperations(repo_path=git_repo)
        ops.create_branch("experiment/test-branch")

        branch = ops.get_current_branch()
        assert branch == "experiment/test-branch"

    @patch("subprocess.run")
    def test_returns_stripped_branch_name(self, mock_run: MagicMock) -> None:
        """The stdout is stripped before returning."""
        ok_result = MagicMock()
        ok_result.returncode = 0
        ok_result.stdout = "experiment/abc123\n"
        ok_result.stderr = ""
        mock_run.return_value = ok_result

        ops = GitOperations(repo_path="/fake/repo")
        branch = ops.get_current_branch()

        assert branch == "experiment/abc123"
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path("/fake/repo").resolve(),
        )

    @patch("subprocess.run")
    def test_raises_runtime_error_on_failure(self, mock_run: MagicMock) -> None:
        """RuntimeError raised when git rev-parse fails (AE-15, T4)."""
        fail_result = MagicMock()
        fail_result.returncode = 128
        fail_result.stderr = "fatal: not a git repository"
        mock_run.return_value = fail_result

        ops = GitOperations(repo_path="/fake/repo")
        with pytest.raises(RuntimeError, match="Failed to get current branch"):
            ops.get_current_branch()
