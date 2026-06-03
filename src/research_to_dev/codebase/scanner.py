"""AST-based scanner for Python codebase analysis.

Pure synchronous functions — no external dependencies, no async.
Discovers ``.py`` files via ``os.walk()``, parses them with
``ast.parse()``, and extracts functions, classes, and methods.
"""

from __future__ import annotations

import ast
import fnmatch
import logging
import os

from research_to_dev.shared.config import CodebaseConfig

_logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def scan_project(
    root: str,
    config: CodebaseConfig,
) -> tuple[list[dict], int]:
    """Discover and scan all ``.py`` files under *root*.

    Walks the directory tree, applies ``include_patterns`` and
    ``exclude_patterns`` from *config*, and calls ``scan_file()``
    for each matching file.

    Args:
        root: Absolute or relative path to the project root.
        config: ``CodebaseConfig`` controlling include/exclude patterns,
            ``max_components_per_module``, and ``include_private``.

    Returns:
        Tuple of ``(all_components, total_files_scanned)``.
        ``all_components`` is the aggregated list of raw component
        dicts from every scanned file.
    """
    all_components: list[dict] = []
    files_scanned = 0

    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            # Include check
            if not any(
                fnmatch.fnmatch(filename, pat)
                for pat in config.include_patterns
            ):
                continue

            # Exclude check
            if any(
                fnmatch.fnmatch(filename, pat)
                for pat in config.exclude_patterns
            ):
                continue

            file_path = os.path.join(dirpath, filename)
            components = scan_file(
                file_path,
                root=root,
                max_components=config.max_components_per_module,
                include_private=config.include_private,
            )
            all_components.extend(components)
            files_scanned += 1

    return all_components, files_scanned


def scan_file(
    file_path: str,
    *,
    root: str = "",
    max_components: int = 200,
    include_private: bool = False,
) -> list[dict]:
    """Parse a single Python file and extract public-component metadata.

    Extracts top-level functions (including ``async def``), classes,
    and class-level methods. Uses ``__all__`` when present for public
    API detection; otherwise filters ``_``-prefixed names unless
    *include_private* is ``True``.

    Args:
        file_path: Absolute or relative path to the ``.py`` file.
        root: Project root for computing ``module_name``. If empty,
            ``module_name`` is derived from *file_path* alone.
        max_components: Hard cap on the number of components returned
            for this file (CA-04).
        include_private: Whether to include ``_``-prefixed names
            when ``__all__`` is not present.

    Returns:
        List of raw component dicts with keys: ``name``, ``kind``,
        ``file_path``, ``line_start``, ``line_end``, ``signature``,
        ``module_name``, ``docstring``.
    """
    module_name = _module_name(file_path, root)

    try:
        with open(file_path, encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        _logger.warning("Skipping file with syntax error: %s — %s", file_path, exc)
        return []
    except Exception as exc:
        _logger.warning("Failed to read file %s: %s", file_path, exc)
        return []

    all_names = _detect_all(tree)
    components: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            comp = _make_component(
                node,
                file_path=file_path,
                module_name=module_name,
            )
            if comp is not None:
                components.append(comp)

        elif isinstance(node, ast.ClassDef):
            comp = _make_component(
                node,
                file_path=file_path,
                module_name=module_name,
            )
            if comp is not None:
                components.append(comp)

            # Extract methods inside the class body
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method = _make_component(
                        child,
                        file_path=file_path,
                        module_name=module_name,
                        parent_class=node.name,
                    )
                    if method is not None:
                        components.append(method)

    # Apply public-API filtering
    if all_names is not None:
        all_set = set(all_names)
        components = [c for c in components if c["name"] in all_set]
    elif not include_private:
        components = [c for c in components if not c["name"].startswith("_")]

    # Enforce component cap per module (CA-04)
    if len(components) > max_components:
        components = components[:max_components]

    return components


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _module_name(file_path: str, root: str) -> str:
    """Derive a dotted module name from *file_path*, relative to *root*.

    >>> _module_name("/proj/src/a/b.py", "/proj")
    'src.a.b'
    """
    if root:
        try:
            rel = os.path.relpath(file_path, root)
        except ValueError:
            rel = file_path
    else:
        rel = file_path

    if rel.endswith(".py"):
        rel = rel[:-3]

    # Strip leading separators to avoid empty first component
    rel = rel.lstrip(os.sep).lstrip("/")

    # Normalize OS separators to dots
    return rel.replace(os.sep, ".").replace("/", ".")


def _detect_all(tree: ast.AST) -> list[str] | None:
    """Try to extract a static ``__all__`` list from an AST module.

    Returns the list of public names as strings, or ``None`` if no
    ``__all__`` assignment was found or the value could not be
    statically evaluated.
    """
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                try:
                    result = ast.literal_eval(node.value)
                    if isinstance(result, list) and all(
                        isinstance(x, str) for x in result
                    ):
                        return result  # type: ignore[return-value]
                except (ValueError, SyntaxError):
                    _logger.debug(
                        "Could not statically evaluate __all__ — "
                        "falling back to prefix filter"
                    )
                    return None
    return None


def _make_component(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    *,
    file_path: str,
    module_name: str,
    parent_class: str | None = None,
) -> dict | None:
    """Build a raw component dict from an AST node.

    Returns ``None`` if the node is not a recognized definition type.
    """
    if isinstance(node, ast.ClassDef):
        kind = "class"
    elif parent_class is not None:
        kind = "method"
    else:
        kind = "function"

    return {
        "name": node.name,
        "kind": kind,
        "file_path": file_path,
        "line_start": node.lineno,
        "line_end": getattr(node, "end_lineno", node.lineno),
        "signature": _signature(node),
        "module_name": module_name,
        "docstring": ast.get_docstring(node),
    }


def _signature(node: ast.AST) -> str:
    """Reconstruct a clean signature from an AST definition node.

    Builds a body-less clone so that ``ast.unparse()`` only emits
    decorators and the definition header. The trailing ``pass``
    line is stripped.

    Position metadata (``lineno``, ``col_offset``, etc.) is copied
    from the original so that ``ast.unparse`` works correctly.
    """
    # Gather fields, replacing 'body' with [Pass()]
    clone_fields: dict[str, object] = {}
    for f in node._fields:
        if f == "body":
            clone_fields[f] = [ast.Pass()]
        else:
            clone_fields[f] = getattr(node, f)

    clone = type(node)(**clone_fields)

    # Copy position metadata required by ast.unparse
    for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
        if hasattr(node, attr):
            setattr(clone, attr, getattr(node, attr))

    text = ast.unparse(clone).strip()

    # Remove the trailing 'pass' that was injected
    lines = text.split("\n")
    if lines and lines[-1].strip() == "pass":
        lines = lines[:-1]

    return "\n".join(lines).strip()
