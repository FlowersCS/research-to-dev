"""Data contracts for the codebase analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Component:
    """A function, class, or method extracted from the codebase via AST.

    The ``description`` field holds text ready for embedding by
    ``CorrelationMap``. It is populated by the ``CodebaseDescriber``
    (LLM-generated when docstring is absent, preserved when present).
    """

    name: str
    """Component name (e.g. ``'train_model'``)."""

    kind: str
    """Component kind: ``'function'``, ``'class'``, or ``'method'``."""

    file_path: str
    """Absolute or relative path to the source file."""

    line_start: int
    """1-based line number where the component definition starts."""

    line_end: int
    """1-based line number where the component definition ends."""

    signature: str
    """Reconstructed signature from ``ast.unparse()`` on the def/class node."""

    description: str
    """Natural language description — ready for embedding by ``CorrelationMap``."""

    module_name: str
    """Dotted module name derived from the file path (e.g. ``'training.pipeline'``)."""

    docstring: str | None = None
    """Original docstring extracted via ``ast.get_docstring()``, or ``None``."""


@dataclass
class ModuleSummary:
    """LLM-generated architectural summary of a Python module.

    Produced by the ``CodebaseDescriber`` after it inspects the full
    source code and component list of a single file.
    """

    module_name: str
    """Dotted module name matching ``Component.module_name``."""

    file_path: str
    """Path to the source file this summary describes."""

    summary: str
    """2–3 sentence architectural description of what the module does."""

    key_responsibilities: list[str] = field(default_factory=list)
    """List of 2–5 key responsibilities this module fulfills."""


@dataclass
class CodebaseContext:
    """Top-level result of a full codebase scan and analysis.

    Consumed by downstream components (``CorrelationMap``,
    ``HypothesisGenerator``). Produced by ``CodebaseAnalysisPipeline``.
    """

    project_name: str
    """Display name of the project (derived from directory name)."""

    project_path: str
    """Absolute path to the project root that was scanned."""

    language: str
    """Programming language identifier (``'python'`` for this MVP)."""

    components: list[Component] = field(default_factory=list)
    """All extracted components across the scanned project."""

    modules: list[ModuleSummary] = field(default_factory=list)
    """Architectural summaries for each scanned module."""

    total_files_scanned: int = 0
    """Number of ``.py`` files discovered and processed."""

    total_components_found: int = 0
    """Total number of components extracted before capping."""


@dataclass
class CodebaseError:
    """Metadata about a failure during codebase scanning or analysis.

    Follows the same pattern as ``RankingError`` and ``ProfilingError``.
    """

    file_path: str
    """Path to the file that caused the error."""

    message: str
    """Human-readable error description."""

    exception_type: str | None = None
    """Python exception class name for debugging."""
