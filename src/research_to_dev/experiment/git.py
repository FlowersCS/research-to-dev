"""Git operations for experiment isolation — dirty check, branch creation,
commit, reset, and branch queries.

Uses ``subprocess`` for zero-dependency git interaction (AD-06).  Fails
loud with actionable error messages for common git failures.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitOperations:
    """Git operations for experiment setup and execution.

    Constructor-injectable so tests can point to temp repos without
    touching the real working tree.

    Usage::

        git_ops = GitOperations(repo_path=".")
        if git_ops.is_dirty():
            raise RuntimeError("Working tree is dirty.")
        git_ops.create_branch("experiment/abc123")
        git_ops.commit_changes("Iteration 1 keep")
        git_ops.reset_hard()
    """

    def __init__(self, repo_path: str = ".") -> None:
        self._repo = Path(repo_path).resolve()

    # -- public API --------------------------------------------------------

    def is_dirty(self) -> bool:
        """Check whether the working tree has uncommitted changes.

        Returns:
            ``True`` if there are tracked or untracked changes.
        """
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=self._repo,
        )
        return bool(result.stdout.strip())

    def is_repo(self) -> bool:
        """Check whether the path is inside a git repository.

        Returns:
            ``True`` if ``git rev-parse --is-inside-work-tree`` succeeds.
        """
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=self._repo,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def branch_exists(self, name: str) -> bool:
        """Check whether a local branch already exists.

        Args:
            name: Branch name (e.g. ``"experiment/abc123"``).

        Returns:
            ``True`` if the branch exists locally.
        """
        result = subprocess.run(
            ["git", "branch", "--list", name],
            capture_output=True,
            text=True,
            cwd=self._repo,
        )
        return bool(result.stdout.strip())

    def create_branch(self, name: str) -> None:
        """Create and switch to a new git branch.

        Args:
            name: Branch name (e.g. ``"experiment/abc123"``).

        Raises:
            RuntimeError: If the working tree is dirty (Gap 3), if the
                branch already exists, or if ``git checkout -b`` fails.
        """
        if self.is_dirty():
            raise RuntimeError(
                "Working tree has uncommitted changes. "
                "Commit or stash before setting up an experiment."
            )

        if self.branch_exists(name):
            raise RuntimeError(
                f"Experiment branch '{name}' already exists."
            )

        result = subprocess.run(
            ["git", "checkout", "-b", name],
            capture_output=True,
            text=True,
            cwd=self._repo,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create branch '{name}': {result.stderr.strip()}"
            )

    def commit_changes(self, message: str) -> None:
        """Stage all changes and commit with *message*.

        Used by the experiment runner to persist improvements (GE-01).

        Args:
            message: The commit message (e.g. ``"Iteration 3 keep"``).

        Raises:
            RuntimeError: If ``git add -A`` or ``git commit`` fails (T4).
        """
        result = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
            text=True,
            cwd=self._repo,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to stage changes: {result.stderr.strip()}"
            )

        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            cwd=self._repo,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to commit changes: {result.stderr.strip()}"
            )

    def reset_hard(self) -> None:
        """Discard all unstaged changes in the working tree.

        Runs ``git checkout .`` — does NOT reset commits or the staging
        area (GE-02).  Used by the experiment runner to revert failed
        iterations.

        Raises:
            RuntimeError: If ``git checkout .`` fails (T4).
        """
        result = subprocess.run(
            ["git", "checkout", "."],
            capture_output=True,
            text=True,
            cwd=self._repo,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to discard working tree changes: {result.stderr.strip()}"
            )

    def checkout(self) -> None:
        """Alias for :meth:`reset_hard` for clarity in keep/discard context.

        Raises:
            RuntimeError: If the underlying ``git checkout .`` fails (T4).
        """
        self.reset_hard()

    def get_current_branch(self) -> str:
        """Return the name of the currently checked-out branch.

        Runs ``git rev-parse --abbrev-ref HEAD`` (GE-04).

        Returns:
            The branch name (e.g. ``"experiment/abc123"``).

        Raises:
            RuntimeError: If the git command fails (T4).
        """
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=self._repo,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to get current branch: {result.stderr.strip()}"
            )
        return result.stdout.strip()
