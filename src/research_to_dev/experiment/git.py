"""Git operations for experiment isolation — dirty check and branch creation.

Uses ``subprocess`` for zero-dependency git interaction (AD-06).  Fails
loud with actionable error messages for common git failures.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitOperations:
    """Git operations for experiment setup — dirty check and branch creation.

    Constructor-injectable so tests can point to temp repos without
    touching the real working tree.

    Usage::

        git_ops = GitOperations(repo_path=".")
        if git_ops.is_dirty():
            raise RuntimeError("Working tree is dirty.")
        git_ops.create_branch("experiment/abc123")
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
