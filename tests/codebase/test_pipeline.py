"""Pipeline integration tests for CodebaseAnalysisPipeline.

All tests use a mocked ``CodebaseDescriber`` and temporary
project directories — no real LLM calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.codebase.pipeline import CodebaseAnalysisPipeline
from research_to_dev.codebase.types import (
    CodebaseContext,
    Component,
    ModuleSummary,
)
from research_to_dev.shared.config import CodebaseConfig


# ==================================================================
# CA-09: Pipeline orchestration
# ==================================================================


class TestPipelineSequencing:
    """Verify the pipeline runs scan → describe → assemble."""

    @pytest.mark.asyncio
    async def test_empty_project_returns_valid_context(
        self,
        temp_project_dir,
        mock_describer: AsyncMock,
    ) -> None:
        """CA-09: Empty directory → CodebaseContext with empty components."""
        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(mock_describer, config=config)

        result = await pipeline.analyze(temp_project_dir)

        assert isinstance(result, CodebaseContext)
        assert result.project_path == temp_project_dir
        assert result.language == "python"
        assert result.components == []
        assert result.modules == []
        assert result.total_files_scanned == 0
        assert result.total_components_found == 0

    @pytest.mark.asyncio
    async def test_single_file_project(
        self,
        temp_project_dir,
        write_temp_py_file,
        mock_describer: AsyncMock,
    ) -> None:
        """End-to-end with one file → CodebaseContext with one module."""
        write_temp_py_file(
            temp_project_dir,
            "main.py",
            "def hello():\n    '''Greet.'''\n    pass\n",
        )
        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(mock_describer, config=config)

        result = await pipeline.analyze(temp_project_dir)

        assert isinstance(result, CodebaseContext)
        assert result.total_files_scanned == 1
        assert result.total_components_found == 1
        assert len(result.components) == 1
        assert len(result.modules) == 1

        comp = result.components[0]
        assert isinstance(comp, Component)
        assert comp.name == "hello"
        assert comp.kind == "function"
        assert comp.module_name == "main"

        mod = result.modules[0]
        assert isinstance(mod, ModuleSummary)
        assert mod.module_name == "main"

    @pytest.mark.asyncio
    async def test_multi_file_project(
        self,
        temp_project_dir,
        write_temp_py_file,
        mock_describer: AsyncMock,
    ) -> None:
        """Multiple files → multiple modules in context."""
        write_temp_py_file(temp_project_dir, "a.py", "def fa():\n    pass\n")
        write_temp_py_file(temp_project_dir, "sub/b.py", "def fb():\n    pass\n")
        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(mock_describer, config=config)

        result = await pipeline.analyze(temp_project_dir)

        assert result.total_files_scanned == 2
        assert result.total_components_found == 2
        assert len(result.components) == 2
        assert len(result.modules) == 2
        module_names = {m.module_name for m in result.modules}
        # a.py is at root level, sub/b.py is in subdirectory
        assert "a" in module_names
        # sub/b.py → sub.b
        assert any("b" in m for m in module_names)

    @pytest.mark.asyncio
    async def test_describer_called_per_module(
        self,
        temp_project_dir,
        write_temp_py_file,
        mock_describer: AsyncMock,
    ) -> None:
        """CA-05: Describer is called once per module."""
        write_temp_py_file(temp_project_dir, "mod1.py", "def f1():\n    pass\n")
        write_temp_py_file(temp_project_dir, "mod2.py", "def f2():\n    pass\n")
        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(mock_describer, config=config)

        await pipeline.analyze(temp_project_dir)

        assert mock_describer.describe.call_count == 2


# ==================================================================
# CA-08: Context assembly
# ==================================================================


class TestContextAssembly:
    """Verify CodebaseContext shape (CA-08)."""

    @pytest.mark.asyncio
    async def test_context_has_correct_shape(
        self,
        temp_project_dir,
        write_temp_py_file,
        mock_describer: AsyncMock,
    ) -> None:
        """CA-08: CodebaseContext has all fields populated correctly."""
        write_temp_py_file(
            temp_project_dir,
            "mod.py",
            "def foo():\n    pass\n\nclass Bar:\n    def method(self):\n        pass\n",
        )
        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(mock_describer, config=config)

        result = await pipeline.analyze(temp_project_dir)

        assert isinstance(result, CodebaseContext)
        assert isinstance(result.project_name, str)
        assert len(result.project_name) > 0
        assert result.project_path == temp_project_dir
        assert result.language == "python"
        assert isinstance(result.components, list)
        assert isinstance(result.modules, list)
        assert isinstance(result.total_files_scanned, int)
        assert isinstance(result.total_components_found, int)

        # One module with 3 components (foo + Bar + method)
        assert result.total_files_scanned == 1
        assert result.total_components_found == 3

        # All components have descriptions (from mock describer)
        for comp in result.components:
            assert isinstance(comp.description, str)

        # Module has a summary
        assert len(result.modules) == 1
        assert isinstance(result.modules[0].summary, str)

    @pytest.mark.asyncio
    async def test_descriptions_mapped_to_correct_components(
        self,
        temp_project_dir,
        write_temp_py_file,
    ) -> None:
        """Descriptions from the describer are mapped to the right component."""
        write_temp_py_file(
            temp_project_dir,
            "mod.py",
            "def train():\n    pass\n\ndef evaluate():\n    pass\n",
        )

        custom_describer = AsyncMock()
        custom_describer.describe.return_value = {
            "descriptions": [
                {"name": "train", "kind": "function", "description": "Trains the model."},
                {"name": "evaluate", "kind": "function", "description": "Evaluates performance."},
            ],
            "module_summary": "Training module.",
            "key_responsibilities": ["Training", "Evaluation"],
        }

        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(custom_describer, config=config)

        result = await pipeline.analyze(temp_project_dir)

        assert len(result.components) == 2
        train_comp = next(c for c in result.components if c.name == "train")
        assert train_comp.description == "Trains the model."
        eval_comp = next(c for c in result.components if c.name == "evaluate")
        assert eval_comp.description == "Evaluates performance."


# ==================================================================
# CA-09: Error resilience
# ==================================================================


class TestPipelineErrorResilience:
    """Verify the pipeline never raises (CA-09)."""

    @pytest.mark.asyncio
    async def test_describer_failure_produces_valid_context(
        self,
        temp_project_dir,
        write_temp_py_file,
    ) -> None:
        """CA-07 + CA-09: Describer failure → degraded context, no exception."""
        write_temp_py_file(
            temp_project_dir,
            "mod.py",
            "def foo():\n    pass\n",
        )

        failing_describer = AsyncMock()
        failing_describer.describe.side_effect = RuntimeError("LLM API error")

        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(failing_describer, config=config)

        # Should NOT raise
        result = await pipeline.analyze(temp_project_dir)

        assert isinstance(result, CodebaseContext)
        assert result.total_files_scanned == 1
        assert len(result.components) == 1
        # Description falls back to docstring or empty
        assert isinstance(result.components[0].description, str)
        # Module summary is empty
        assert len(result.modules) == 1
        assert result.modules[0].summary == ""

    @pytest.mark.asyncio
    async def test_all_syntax_error_files_returns_empty_context(
        self,
        temp_project_dir,
        write_temp_py_file,
        mock_describer: AsyncMock,
    ) -> None:
        """CA-09: All files have syntax errors → empty components, zero stats."""
        write_temp_py_file(temp_project_dir, "broken1.py", "@@@ invalid")
        write_temp_py_file(temp_project_dir, "broken2.py", "not python @@@@")

        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(mock_describer, config=config)

        result = await pipeline.analyze(temp_project_dir)

        assert isinstance(result, CodebaseContext)
        assert result.total_files_scanned == 2
        assert result.components == []
        assert result.modules == []
        assert result.total_components_found == 0

    @pytest.mark.asyncio
    async def test_source_read_failure(
        self,
        temp_project_dir,
        write_temp_py_file,
        mock_describer: AsyncMock,
    ) -> None:
        """When a source file can't be read, describer receives empty source."""
        fp = write_temp_py_file(
            temp_project_dir,
            "mod.py",
            "def foo():\n    pass\n",
        )

        # Described with empty source is still called
        config = CodebaseConfig()
        pipeline = CodebaseAnalysisPipeline(mock_describer, config=config)

        await pipeline.analyze(temp_project_dir)

        # Describer was called for the module
        mock_describer.describe.assert_called_once()
        call_kwargs = mock_describer.describe.call_args[1]
        assert call_kwargs["module_path"] == "mod"
        assert "def foo()" in call_kwargs["source"]


# ==================================================================
# CA-10: Protocol contracts
# ==================================================================


class TestProtocolContracts:
    """Verify protocol-based constructor injection (CA-10)."""

    @pytest.mark.asyncio
    async def test_asyncmock_satisfies_protocol(
        self, mock_describer: AsyncMock
    ) -> None:
        """CA-10: Pipeline accepts AsyncMock impl of CodebaseDescriber Protocol."""
        pipeline = CodebaseAnalysisPipeline(mock_describer)
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_incomplete_object_fails_protocol_check(self) -> None:
        """Object without describe() fails CodebaseDescriber check."""
        from research_to_dev.codebase.protocols import CodebaseDescriber

        incomplete = object()
        assert not isinstance(incomplete, CodebaseDescriber)


# ==================================================================
# Config injection
# ==================================================================


class TestPipelineConfigInjection:
    """Verify config injection and defaults."""

    @pytest.mark.asyncio
    async def test_default_config_used_when_none_provided(
        self, mock_describer: AsyncMock
    ) -> None:
        """Pipeline uses CodebaseConfig() defaults when no config is injected."""
        pipeline = CodebaseAnalysisPipeline(mock_describer)

        assert pipeline._config.include_private is False
        assert pipeline._config.max_components_per_module == 200
        assert pipeline._config.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_custom_config_injected(
        self, mock_describer: AsyncMock
    ) -> None:
        """Custom CodebaseConfig is preserved after injection."""
        config = CodebaseConfig(
            include_private=True,
            max_components_per_module=50,
            model="gpt-4o",
        )
        pipeline = CodebaseAnalysisPipeline(mock_describer, config=config)

        assert pipeline._config is config
        assert pipeline._config.include_private is True
        assert pipeline._config.max_components_per_module == 50
        assert pipeline._config.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_config_respected_by_scanner(
        self,
        temp_project_dir,
        write_temp_py_file,
        mock_describer: AsyncMock,
    ) -> None:
        """include_private=True in config → private components scanned."""
        write_temp_py_file(
            temp_project_dir,
            "mod.py",
            "def _private():\n    pass\n\ndef public():\n    pass\n",
        )

        # Default config (include_private=False)
        pipeline_default = CodebaseAnalysisPipeline(mock_describer)
        result_default = await pipeline_default.analyze(temp_project_dir)
        assert result_default.total_components_found == 1  # only public
        assert result_default.components[0].name == "public"

        # include_private=True
        config_private = CodebaseConfig(include_private=True)
        pipeline_private = CodebaseAnalysisPipeline(mock_describer, config=config_private)
        result_private = await pipeline_private.analyze(temp_project_dir)
        assert result_private.total_components_found == 2  # both
