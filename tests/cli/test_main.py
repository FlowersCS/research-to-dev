"""CLI tests for the research-to-dev codebase analyze command.

Uses typer.testing.CliRunner for invocation without subprocess overhead.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ==================================================================
# Command invocation
# ==================================================================


class TestCodebaseAnalyzeCommand:
    """Verify the 'codebase analyze' command."""

    def test_help_output(self) -> None:
        """'codebase analyze --help' shows usage."""
        from research_to_dev.cli.main import app

        result = runner.invoke(app, ["codebase", "analyze", "--help"])
        assert result.exit_code == 0
        assert "Analyze a Python codebase" in result.stdout

    def test_parent_help_output(self) -> None:
        """'codebase --help' shows subcommands."""
        from research_to_dev.cli.main import app

        result = runner.invoke(app, ["codebase", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.stdout

    def test_root_help_output(self) -> None:
        """'--help' shows top-level commands."""
        from research_to_dev.cli.main import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "codebase" in result.stdout

    def test_invalid_path_shows_error(self) -> None:
        """Passing a non-existent directory → exit code 1."""
        from research_to_dev.cli.main import app

        result = runner.invoke(app, ["codebase", "analyze", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "Error:" in result.stdout or "Error:" in result.stderr

    def test_default_path_is_current_directory(self) -> None:
        """No path argument defaults to current directory."""
        import os

        from research_to_dev.cli.main import app

        with (
            patch(
                "research_to_dev.cli.main.AsyncOpenAI",
                return_value=AsyncMock(),
            ),
            patch(
                "research_to_dev.cli.main.CodebaseAnalysisPipeline.analyze",
                new_callable=AsyncMock,
            ) as mock_analyze,
        ):
            from research_to_dev.codebase.types import CodebaseContext

            mock_analyze.return_value = CodebaseContext(
                project_name="test-proj",
                project_path=os.getcwd(),
                language="python",
            )

            result = runner.invoke(app, ["codebase", "analyze"])
            assert result.exit_code == 0
            mock_analyze.assert_called_once()
            # Verify the path argument was the current directory
            call_path = mock_analyze.call_args[0][0]
            assert os.path.isabs(call_path)

    def test_path_argument_is_respected(self) -> None:
        """Passing a specific path uses that path."""
        import os

        from research_to_dev.cli.main import app

        with (
            patch(
                "research_to_dev.cli.main.AsyncOpenAI",
                return_value=AsyncMock(),
            ),
            patch(
                "research_to_dev.cli.main.CodebaseAnalysisPipeline.analyze",
                new_callable=AsyncMock,
            ) as mock_analyze,
        ):
            from research_to_dev.codebase.types import CodebaseContext

            mock_analyze.return_value = CodebaseContext(
                project_name="test",
                project_path="/tmp/test_project",
                language="python",
            )

            result = runner.invoke(app, ["codebase", "analyze", "/tmp"])
            assert result.exit_code == 0
            call_path = mock_analyze.call_args[0][0]
            assert call_path == "/tmp"

    def test_cli_options_are_passed_to_config(self) -> None:
        """--include-private and --max-components create correct config."""
        import os

        from research_to_dev.cli.main import app

        with (
            patch(
                "research_to_dev.cli.main.AsyncOpenAI",
                return_value=AsyncMock(),
            ),
            patch(
                "research_to_dev.cli.main.CodebaseAnalysisPipeline.__init__",
                return_value=None,
            ) as mock_init,
            patch(
                "research_to_dev.cli.main.CodebaseAnalysisPipeline.analyze",
                new_callable=AsyncMock,
            ) as mock_analyze,
        ):
            from research_to_dev.codebase.types import CodebaseContext

            mock_analyze.return_value = CodebaseContext(
                project_name="test",
                project_path=os.getcwd(),
                language="python",
            )

            result = runner.invoke(
                app,
                [
                    "codebase",
                    "analyze",
                    "--include-private",
                    "--max-components",
                    "50",
                    "--model",
                    "gpt-4o",
                ],
            )
            assert result.exit_code == 0

            # Verify config was injected with correct values
            call_kwargs = mock_init.call_args[1]
            config = call_kwargs["config"]
            assert config.include_private is True
            assert config.max_components_per_module == 50
            assert config.model == "gpt-4o"

    def test_output_contains_summary(self) -> None:
        """Output includes project name, files scanned, components found."""
        import os

        from research_to_dev.cli.main import app

        with (
            patch(
                "research_to_dev.cli.main.AsyncOpenAI",
                return_value=AsyncMock(),
            ),
            patch(
                "research_to_dev.cli.main.CodebaseAnalysisPipeline.analyze",
                new_callable=AsyncMock,
            ) as mock_analyze,
        ):
            from research_to_dev.codebase.types import CodebaseContext, ModuleSummary

            mock_analyze.return_value = CodebaseContext(
                project_name="demo",
                project_path="/tmp",
                language="python",
                components=[],
                modules=[
                    ModuleSummary(
                        module_name="main",
                        file_path="/tmp/main.py",
                        summary="Entry point module.",
                        key_responsibilities=["CLI entry"],
                    ),
                ],
                total_files_scanned=3,
                total_components_found=10,
            )

            result = runner.invoke(app, ["codebase", "analyze", "/tmp"])
            assert result.exit_code == 0
            assert "Project: demo" in result.stdout
            assert "Files scanned: 3" in result.stdout
            assert "Components found: 10" in result.stdout
            assert "Modules described: 1" in result.stdout
            assert "[main]" in result.stdout
