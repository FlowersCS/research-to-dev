"""Unit tests for the AST-based Python codebase scanner.

Covers file discovery, component extraction, ``__all__`` filtering,
private-name filtering, component capping, and syntax-error resilience.
All tests use temporary ``.py`` files — no real projects needed.
"""

from __future__ import annotations

import os

from research_to_dev.codebase.scanner import (
    _detect_all,
    _module_name,
    _signature,
    scan_file,
    scan_project,
)
from research_to_dev.shared.config import CodebaseConfig


# ==================================================================
# _module_name helper
# ==================================================================


class TestModuleName:
    """Verify dotted module name derivation from file paths."""

    def test_relative_to_root(self) -> None:
        """Converts file path relative to root to dotted module name."""
        result = _module_name("/proj/src/a/b.py", "/proj")
        assert result == "src.a.b"

    def test_same_as_root(self) -> None:
        """File directly at root produces simple name."""
        result = _module_name("/proj/main.py", "/proj")
        assert result == "main"

    def test_no_root_uses_full_path(self) -> None:
        """Without root, the full stripped path is used."""
        result = _module_name("/some/path/module.py", "")
        assert result == "some.path.module"

    def test_strips_py_extension(self) -> None:
        """The .py suffix is always removed."""
        result = _module_name("foo/bar.py", "")
        assert result == "foo.bar"

    def test_non_py_files_unchanged(self) -> None:
        """Non-.py extensions are preserved (shouldn't happen in practice)."""
        result = _module_name("foo/bar.txt", "")
        assert result == "foo.bar.txt"


# ==================================================================
# _signature helper
# ==================================================================


class TestSignature:
    """Verify signature reconstruction via ast.unparse()."""

    def test_simple_function(self) -> None:
        """Reconstructs a basic function signature."""
        import ast

        code = "def foo(x: int) -> str:\n    return str(x)"
        tree = ast.parse(code)
        node = tree.body[0]
        sig = _signature(node)
        assert "def foo(x: int) -> str:" in sig
        assert "pass" not in sig

    def test_decorated_function(self) -> None:
        """Decorators are included in the signature."""
        import ast

        code = "@staticmethod\ndef bar() -> None:\n    pass"
        tree = ast.parse(code)
        node = tree.body[0]
        sig = _signature(node)
        assert "@staticmethod" in sig
        assert "def bar() -> None:" in sig

    def test_class_definition(self) -> None:
        """Class signature includes bases."""
        import ast

        code = "class MyClass(BaseClass):\n    pass"
        tree = ast.parse(code)
        node = tree.body[0]
        sig = _signature(node)
        assert "class MyClass(BaseClass):" in sig

    def test_async_function(self) -> None:
        """Async functions are correctly identified."""
        import ast

        code = "async def fetch(url: str) -> dict:\n    return {}"
        tree = ast.parse(code)
        node = tree.body[0]
        sig = _signature(node)
        assert "async def fetch(url: str) -> dict:" in sig


# ==================================================================
# _detect_all helper
# ==================================================================


class TestDetectAll:
    """Verify __all__ extraction from AST."""

    def test_detects_static_list(self) -> None:
        """A simple __all__ = ['a', 'b'] is extracted."""
        import ast

        code = "__all__ = ['public_fn', 'PublicClass']\n\ndef public_fn():\n    pass"
        tree = ast.parse(code)
        result = _detect_all(tree)
        assert result == ["public_fn", "PublicClass"]

    def test_returns_none_when_no_all(self) -> None:
        """Returns None when __all__ is not defined."""
        import ast

        code = "def foo():\n    pass"
        tree = ast.parse(code)
        result = _detect_all(tree)
        assert result is None

    def test_returns_none_for_dynamic_all(self) -> None:
        """Returns None for dynamically computed __all__."""
        import ast

        code = "__all__ = [n for n in dir() if not n.startswith('_')]"
        tree = ast.parse(code)
        result = _detect_all(tree)
        assert result is None

    def test_returns_none_for_non_list_all(self) -> None:
        """Returns None if __all__ is not a list of strings."""
        import ast

        code = "__all__ = 42"
        tree = ast.parse(code)
        result = _detect_all(tree)
        assert result is None


# ==================================================================
# scan_file — CA-01, CA-02
# ==================================================================


class TestScanFile:
    """Verify scan_file() component extraction."""

    def test_empty_file_returns_empty(self, temp_project_dir, write_temp_py_file) -> None:
        """An empty .py file produces no components."""
        fp = write_temp_py_file(temp_project_dir, "empty.py", "")
        result = scan_file(fp, root=temp_project_dir)
        assert result == []

    def test_only_comments_and_imports(self, temp_project_dir, write_temp_py_file) -> None:
        """A file with only comments and imports produces no components."""
        fp = write_temp_py_file(
            temp_project_dir,
            "imports.py",
            "# just a comment\nimport os\nfrom pathlib import Path\n",
        )
        result = scan_file(fp, root=temp_project_dir)
        assert result == []

    def test_function_extraction(self, temp_project_dir, write_temp_py_file) -> None:
        """Top-level functions are extracted with correct kind and signature."""
        fp = write_temp_py_file(
            temp_project_dir,
            "funcs.py",
            'def greet(name: str) -> str:\n    """Say hello."""\n    return f"Hello {name}"\n',
        )
        result = scan_file(fp, root=temp_project_dir)
        assert len(result) == 1
        assert result[0]["name"] == "greet"
        assert result[0]["kind"] == "function"
        assert result[0]["signature"] == 'def greet(name: str) -> str:'
        assert result[0]["docstring"] == "Say hello."
        assert result[0]["module_name"] == "funcs"
        assert result[0]["line_start"] == 1
        assert result[0]["line_end"] >= 1

    def test_class_extraction(self, temp_project_dir, write_temp_py_file) -> None:
        """Classes are extracted with kind='class'."""
        fp = write_temp_py_file(
            temp_project_dir,
            "model.py",
            "class Model:\n    pass\n",
        )
        result = scan_file(fp, root=temp_project_dir)
        assert len(result) == 1
        assert result[0]["name"] == "Model"
        assert result[0]["kind"] == "class"

    def test_class_methods_are_included(self, temp_project_dir, write_temp_py_file) -> None:
        """Methods inside a class are extracted with kind='method'."""
        fp = write_temp_py_file(
            temp_project_dir,
            "trainer.py",
            "class Trainer:\n    def fit(self, X, y):\n        pass\n\n    def predict(self, X):\n        pass\n",
        )
        result = scan_file(fp, root=temp_project_dir)
        assert len(result) == 3  # Trainer class + 2 methods
        classes = [c for c in result if c["kind"] == "class"]
        methods = [c for c in result if c["kind"] == "method"]
        assert len(classes) == 1
        assert len(methods) == 2
        assert {m["name"] for m in methods} == {"fit", "predict"}

    def test_mixed_module(self, temp_project_dir, write_temp_py_file) -> None:
        """CA-02: One class + two functions = three components."""
        fp = write_temp_py_file(
            temp_project_dir,
            "mixed.py",
            'def func_one():\n    """First."""\n    pass\n\n'
            "class MyClass:\n    pass\n\n"
            'def func_two(x: int) -> bool:\n    """Second."""\n    return True\n',
        )
        result = scan_file(fp, root=temp_project_dir)
        assert len(result) == 3
        kinds = {c["kind"] for c in result}
        assert kinds == {"function", "class"}

    def test_async_function_extraction(self, temp_project_dir, write_temp_py_file) -> None:
        """Async functions are extracted alongside regular ones."""
        fp = write_temp_py_file(
            temp_project_dir,
            "async_mod.py",
            "async def fetch():\n    pass\n\ndef sync_fn():\n    pass\n",
        )
        result = scan_file(fp, root=temp_project_dir)
        assert len(result) == 2
        assert result[0]["name"] == "fetch"
        assert result[1]["name"] == "sync_fn"

    def test_docstring_none_when_absent(self, temp_project_dir, write_temp_py_file) -> None:
        """Components without docstrings have docstring=None."""
        fp = write_temp_py_file(
            temp_project_dir,
            "nodoc.py",
            "def no_doc():\n    x = 1\n    return x\n",
        )
        result = scan_file(fp, root=temp_project_dir)
        assert result[0]["docstring"] is None

    def test_line_numbers_are_correct(self, temp_project_dir, write_temp_py_file) -> None:
        """line_start and line_end reflect actual positions."""
        fp = write_temp_py_file(
            temp_project_dir,
            "lines.py",
            "# comment\n# comment\n\ndef third_line_fn():\n    pass\n",
        )
        result = scan_file(fp, root=temp_project_dir)
        assert result[0]["line_start"] == 4


# ==================================================================
# CA-03: Public API filtering
# ==================================================================


class TestPublicApiFiltering:
    """Verify __all__ and _ prefix filtering (CA-03)."""

    def test_all_defined_filters_components(self, temp_project_dir, write_temp_py_file) -> None:
        """CA-03: __all__ = ['public_fn'] → only public_fn emitted."""
        fp = write_temp_py_file(
            temp_project_dir,
            "with_all.py",
            "__all__ = ['public_fn']\n\ndef public_fn():\n    pass\n\ndef _private_fn():\n    pass\n",
        )
        result = scan_file(fp, root=temp_project_dir, include_private=False)
        names = {c["name"] for c in result}
        assert names == {"public_fn"}

    def test_no_all_filters_underscore_prefix(self, temp_project_dir, write_temp_py_file) -> None:
        """CA-03: No __all__, include_private=False → _ prefixed filtered out."""
        fp = write_temp_py_file(
            temp_project_dir,
            "no_all.py",
            "def _helper():\n    pass\n\ndef public_fn():\n    pass\n",
        )
        result = scan_file(fp, root=temp_project_dir, include_private=False)
        names = {c["name"] for c in result}
        assert names == {"public_fn"}

    def test_include_private_overrides_underscore_filter(self, temp_project_dir, write_temp_py_file) -> None:
        """include_private=True → _ prefixed names are included."""
        fp = write_temp_py_file(
            temp_project_dir,
            "private_ok.py",
            "def _internal():\n    pass\n\ndef public():\n    pass\n",
        )
        result = scan_file(fp, root=temp_project_dir, include_private=True)
        names = {c["name"] for c in result}
        assert names == {"_internal", "public"}

    def test_all_takes_precedence_over_include_private(self, temp_project_dir, write_temp_py_file) -> None:
        """__all__ overrides include_private — only __all__ items emitted."""
        fp = write_temp_py_file(
            temp_project_dir,
            "all_priority.py",
            "__all__ = ['only_this']\n\ndef only_this():\n    pass\n\ndef _hidden():\n    pass\n\ndef also_hidden():\n    pass\n",
        )
        result = scan_file(fp, root=temp_project_dir, include_private=True)
        names = {c["name"] for c in result}
        assert names == {"only_this"}


# ==================================================================
# CA-04: Component cap
# ==================================================================


class TestComponentCap:
    """Verify max_components_per_module enforcement (CA-04)."""

    def test_enforces_cap(self, temp_project_dir, write_temp_py_file) -> None:
        """CA-04: 20 components with max_components=5 → exactly 5 emitted."""
        funcs = "\n".join(f"def f{i}():\n    pass\n" for i in range(20))
        fp = write_temp_py_file(temp_project_dir, "large.py", funcs)
        result = scan_file(fp, root=temp_project_dir, max_components=5)
        assert len(result) == 5

    def test_cap_below_actual_count_limits(self, temp_project_dir, write_temp_py_file) -> None:
        """When count < cap, all components are returned."""
        fp = write_temp_py_file(
            temp_project_dir, "small.py", "def a():\n    pass\n\ndef b():\n    pass\n"
        )
        result = scan_file(fp, root=temp_project_dir, max_components=100)
        assert len(result) == 2


# ==================================================================
# Syntax error resilience (CA-02 scenario)
# ==================================================================


class TestSyntaxErrorResilience:
    """Verify scanner survives syntax errors gracefully."""

    def test_syntax_error_returns_empty(self, temp_project_dir, write_temp_py_file) -> None:
        """CA-02: Invalid syntax → empty list, no exception raised."""
        fp = write_temp_py_file(
            temp_project_dir,
            "broken.py",
            "def foo(:\n    pass\n",  # missing closing paren
        )
        result = scan_file(fp, root=temp_project_dir)
        assert result == []

    def test_mixed_valid_and_invalid_files(self, temp_project_dir, write_temp_py_file) -> None:
        """Valid files are still scanned after encountering a syntax error."""
        write_temp_py_file(
            temp_project_dir,
            "broken.py",
            "this is not valid python @@@",
        )
        fp_valid = write_temp_py_file(
            temp_project_dir,
            "valid.py",
            "def ok():\n    pass\n",
        )
        result = scan_file(fp_valid, root=temp_project_dir)
        assert len(result) == 1
        assert result[0]["name"] == "ok"


# ==================================================================
# scan_project integration
# ==================================================================


class TestScanProject:
    """Verify scan_project() file discovery and aggregation."""

    def test_discovers_multiple_files(self, temp_project_dir, write_temp_py_file) -> None:
        """Discovers and scans multiple .py files recursively."""
        write_temp_py_file(temp_project_dir, "a.py", "def fa():\n    pass\n")
        write_temp_py_file(temp_project_dir, "sub/b.py", "def fb():\n    pass\n")
        config = CodebaseConfig()

        components, files_scanned = scan_project(temp_project_dir, config)
        assert files_scanned == 2
        assert len(components) == 2
        names = {c["name"] for c in components}
        assert names == {"fa", "fb"}

    def test_respects_exclude_patterns(self, temp_project_dir, write_temp_py_file) -> None:
        """Files matching exclude_patterns are skipped."""
        write_temp_py_file(temp_project_dir, "main.py", "def main():\n    pass\n")
        write_temp_py_file(temp_project_dir, "test_main.py", "def test():\n    pass\n")
        config = CodebaseConfig(
            exclude_patterns=["test_*.py", "*_test.py", "setup.py", "conftest.py"]
        )

        components, files_scanned = scan_project(temp_project_dir, config)
        assert files_scanned == 1
        assert components[0]["name"] == "main"

    def test_respects_include_patterns(self, temp_project_dir, write_temp_py_file) -> None:
        """Only files matching include_patterns are scanned."""
        write_temp_py_file(temp_project_dir, "main.py", "def f():\n    pass\n")
        write_temp_py_file(temp_project_dir, "notes.txt", "not a python file")
        config = CodebaseConfig(include_patterns=["*.py"])

        components, files_scanned = scan_project(temp_project_dir, config)
        assert files_scanned == 1

    def test_empty_directory(self, temp_project_dir) -> None:
        """CA-01: Empty directory → zero files discovered."""
        config = CodebaseConfig()
        components, files_scanned = scan_project(temp_project_dir, config)
        assert files_scanned == 0
        assert components == []

    def test_syntax_error_in_project_continues(self, temp_project_dir, write_temp_py_file) -> None:
        """When one file has a syntax error, others are still scanned."""
        write_temp_py_file(temp_project_dir, "broken.py", "@@@ invalid")
        write_temp_py_file(temp_project_dir, "ok.py", "def f():\n    pass\n")
        config = CodebaseConfig()

        components, files_scanned = scan_project(temp_project_dir, config)
        assert files_scanned == 2  # Both "scanned" even if broken returns empty
        assert len(components) == 1
        assert components[0]["name"] == "f"
