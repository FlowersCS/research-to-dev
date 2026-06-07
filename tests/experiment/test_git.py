"""Tests for GitOperations — dirty check and branch creation.

Uses real git repos in temp directories for integration-level testing.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

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
