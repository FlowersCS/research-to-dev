"""Traceability — connects experimental results back to academic origins
via pipeline trace metadata.

Module follows Screaming Architecture: dataclasses, Protocol, adapter,
and pure-function builders all live in this single module.

Public API:
    CorrelationTrace, HypothesisOrigin, TraceabilityContext
    TraceReader (Protocol)
    build_traceability, JsonTraceReader
    _hash8, _slugify (for anchor generation)

Design decisions:
    D1: New report/traceability.py (Screaming Architecture)
    D2: Anchors use <a id="..."></a> (GFM compatible)
    D7: TraceabilityContext is optional sidecar on CompiledReport
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ===================================================================
# Traceability Dataclasses (TR-01)
# ===================================================================


@dataclass
class CorrelationTrace:
    """A single resolved correlation for traceability.

    Maps a hypothesis correlation to a specific code component or module,
    with the academic paper claim that motivated it.
    """

    correlation_id: str
    """12-char synthetic SHA-256 hash identifying this correlation."""

    paper_title: str
    """Title of the academic paper the claim comes from."""

    claim_text: str
    """Full text of the paper claim that motivated this code change."""

    claim_section: str
    """Section of the paper where the claim appears (e.g. 'methods', 'results')."""

    claim_paper_id: str
    """Identifier of the paper (e.g. arxiv ID)."""

    component_name: str
    """Name of the code component (target_name from correlation)."""

    component_file_path: str
    """Filesystem path to the code component. Empty for ModuleCorrelation."""

    component_module_name: str
    """Python module name of the code component."""

    component_signature: str
    """Full AST signature or module summary."""

    correlation_type: str
    """Type of correlation: direct_solution, related_technique, contradicts, prerequisite."""

    reasoning: str
    """Explanation of why the claim correlates with the code."""

    target_type: str
    """"component" or "module" — what the correlation targets."""


@dataclass
class HypothesisOrigin:
    """Original hypothesis metadata for traceability enrichment.

    Connects a hypothesis to its supporting academic papers and resolved
    code correlations, providing the navigable traceability chain:
    resultado → hipótesis → correlación → paper → claim → code component.
    """

    hypothesis_id: str
    """Identifier for this hypothesis."""

    title: str
    """Title of the hypothesis."""

    description: str
    """Full description of the hypothesis."""

    code_changes: str
    """Description of code changes the hypothesis proposes."""

    supporting_papers: list[str] = field(default_factory=list)
    """List of paper IDs supporting this hypothesis."""

    resolved_correlations: list[CorrelationTrace] = field(default_factory=list)
    """Correlations that were successfully resolved."""

    unresolved_correlation_ids: list[str] = field(default_factory=list)
    """Synthetic correlation IDs that could not be resolved."""


@dataclass
class TraceabilityContext:
    """Optional sidecar on CompiledReport holding traceability data.

    Contains hypothesis origins, claim anchors for the appendix, and
    component anchors for cross-referencing.
    """

    hypothesis_origins: dict[str, HypothesisOrigin] = field(default_factory=dict)
    """Hypothesis ID → HypothesisOrigin mapping."""

    claim_anchors: dict[str, str] = field(default_factory=dict)
    """Anchor ID → claim_text mapping for the appendix."""

    component_anchors: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    """Anchor ID → (file_path, module_name, signature) mapping for the appendix."""


# ===================================================================
# TraceReader Protocol (TR-02)
# ===================================================================


class TraceReader(Protocol):
    """Protocol for reading pipeline trace data into TraceabilityContext.

    Enables test injection and alternative trace sources without
    coupling to a specific file format or filesystem.
    """

    def read(self, trace_path: str) -> TraceabilityContext | None:
        """Read trace data from *trace_path* and build a TraceabilityContext.

        Returns None when trace data is unavailable or unreadable.
        """
        ...


# ===================================================================
# Anchor Generation Helpers (TR-03)
# ===================================================================


def _hash8(text: str) -> str:
    """Return the first 8 characters of the SHA-256 hex digest of *text*.

    Deterministic: same input always produces the same 8-char hex string.
    Used for generating short, unique anchor IDs from claim text or
    component signatures.

    Args:
        text: Any string to hash.

    Returns:
        8-character lowercase hexadecimal string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _slugify(text: str) -> str:
    """Convert *text* to a valid markdown anchor fragment.

    Rules:
    - Lowercase all characters
    - Replace non-alphanumeric characters (except hyphens and underscores)
      with hyphens
    - Collapse consecutive hyphens into one
    - Strip leading and trailing hyphens
    - Return ``"unknown"`` if result is empty

    Args:
        text: The module or component name to slugify.

    Returns:
        A valid markdown fragment identifier.
    """
    if not text:
        return "unknown"

    # Lowercase
    slug = text.lower()
    # Replace anything that isn't alphanumeric, hyphen, or underscore with hyphen
    slug = re.sub(r"[^a-z0-9_-]", "-", slug)
    # Collapse consecutive hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")

    return slug if slug else "unknown"


# ===================================================================
# Correlation Resolution Helpers (TR-04)
# ===================================================================


def _build_correlation_lookup(correlation_dict: dict) -> dict[str, dict]:
    """Build a lookup index from correlation entries keyed by synthetic ID.

    The synthetic ID is computed as ``_hash8(claim.text + target_name)``
    matching the pipeline's CorrelationMap hash scheme.  This allows
    resolving Hypothesis correlation references (which store synthetic
    IDs) back to the full CorrelationMap entry.

    Args:
        correlation_dict: Raw correlation entries from trace.json,
            keyed by correlation ID (not synthetic).

    Returns:
        Dict mapping synthetic hash8 IDs → full correlation entry dict.
    """
    lookup: dict[str, dict] = {}
    for corr_id, entry in correlation_dict.items():
        if not isinstance(entry, dict):
            continue
        claim = entry.get("claim", {})
        claim_text = claim.get("text", "") if isinstance(claim, dict) else ""
        target_name = entry.get("target_name", "")
        synthetic_id = _hash8(claim_text + target_name)
        lookup[synthetic_id] = entry
    return lookup


def _resolve_correlation(
    synthetic_id: str,
    lookup: dict[str, dict],
) -> tuple[dict | None, str]:
    """Resolve a synthetic correlation ID against the lookup index.

    Args:
        synthetic_id: 8-char hex hash from the hypothesis correlation list.
        lookup: Index built by _build_correlation_lookup().

    Returns:
        Tuple of ``(entry_dict, status)`` where status is ``"resolved"``
        or ``"unresolved"``.  Entry is None when unresolved.
    """
    entry = lookup.get(synthetic_id)
    if entry is not None:
        return (entry, "resolved")
    return (None, "unresolved")


# ===================================================================
# Traceability Builder (TR-05)
# ===================================================================


def build_traceability(
    hypothesis_dicts: list[dict],
    correlation_dict: dict | None,
) -> TraceabilityContext:
    """Build a TraceabilityContext from raw pipeline trace data.

    Resolves Hypothesis correlation references (12-char synthetic SHA-256
    IDs) against the CorrelationMap entries by recomputing the hash from
    ``claim.text + target_name``.  Populates both per-hypothesis origins
    and global anchor maps for later Markdown/JSON rendering.

    Error resilience (T1–T4):
    - T2: Unresolved correlation IDs are tracked, not fatal.
    - T3: Empty correlation dict → all unresolved, empty anchors.
    - T4: Malformed correlation entries skipped with logger warning.

    Args:
        hypothesis_dicts: List of raw hypothesis dicts from trace.json
            (``data["hypothesis"]["hypotheses"]``).
        correlation_dict: Raw correlation entries from trace.json
            (``data["correlation"]``).  May be None.

    Returns:
        Populated TraceabilityContext (never None — even empty data
        produces a valid empty context).
    """
    # Build correlation lookup if data is available
    lookup: dict[str, dict] = {}
    if correlation_dict is not None:
        lookup = _build_correlation_lookup(correlation_dict)

    hypothesis_origins: dict[str, HypothesisOrigin] = {}
    claim_anchors: dict[str, str] = {}
    component_anchors: dict[str, tuple[str, str, str]] = {}
    # Track collision counters per anchor prefix
    claim_counters: dict[str, int] = {}
    comp_counters: dict[str, int] = {}

    for hd in hypothesis_dicts:
        if not isinstance(hd, dict):
            continue

        hypo_id = hd.get("id", "")
        if not hypo_id:
            continue

        # --- Resolve correlations ---
        resolved: list[CorrelationTrace] = []
        unresolved: list[str] = []

        correlation_ids: list[str] = hd.get("correlations", [])
        if not isinstance(correlation_ids, list):
            correlation_ids = []

        for synth_id in correlation_ids:
            if not isinstance(synth_id, str):
                continue
            entry, status = _resolve_correlation(synth_id, lookup)
            if status == "resolved" and entry is not None:
                try:
                    ct = _dict_to_correlation_trace(entry, synth_id)
                    resolved.append(ct)

                    # --- Populate claim anchor ---
                    claim_text = ct.claim_text
                    base_anchor = _hash8(claim_text)
                    anchor = _unique_anchor(base_anchor, claim_counters)
                    claim_anchors[anchor] = claim_text

                    # --- Populate component anchor ---
                    comp_base = _slugify(ct.component_module_name)
                    comp_hash = _hash8(ct.component_module_name + ct.component_signature)
                    comp_base_anchor = f"comp-{comp_base}-{comp_hash}"
                    comp_anchor = _unique_anchor(comp_base_anchor, comp_counters)
                    component_anchors[comp_anchor] = (
                        ct.component_file_path,
                        ct.component_module_name,
                        ct.component_signature,
                    )
                except (KeyError, TypeError):
                    # T4: malformed correlation entry — skip, continue
                    unresolved.append(synth_id)
            else:
                unresolved.append(synth_id)

        hypothesis_origins[hypo_id] = HypothesisOrigin(
            hypothesis_id=hypo_id,
            title=hd.get("title", ""),
            description=hd.get("description", ""),
            code_changes=hd.get("code_changes", ""),
            supporting_papers=(
                hd.get("supporting_papers", [])
                if isinstance(hd.get("supporting_papers"), list)
                else []
            ),
            resolved_correlations=resolved,
            unresolved_correlation_ids=unresolved,
        )

    return TraceabilityContext(
        hypothesis_origins=hypothesis_origins,
        claim_anchors=claim_anchors,
        component_anchors=component_anchors,
    )


# ===================================================================
# Private helpers for build_traceability
# ===================================================================


def _dict_to_correlation_trace(entry: dict, correlation_id: str) -> CorrelationTrace:
    """Convert a raw correlation entry dict to a CorrelationTrace dataclass.

    Handles missing keys gracefully with empty-string defaults (T4 resilience).
    """
    claim = entry.get("claim", {})
    claim_text = claim.get("text", "") if isinstance(claim, dict) else ""

    return CorrelationTrace(
        correlation_id=correlation_id,
        paper_title=entry.get("paper_title", ""),
        claim_text=claim_text,
        claim_section=entry.get("claim_section", ""),
        claim_paper_id=entry.get("claim_paper_id", ""),
        component_name=entry.get("component_name", ""),
        component_file_path=entry.get("component_file_path", ""),
        component_module_name=entry.get("component_module_name", ""),
        component_signature=entry.get("component_signature", ""),
        correlation_type=entry.get("correlation_type", ""),
        reasoning=entry.get("reasoning", ""),
        target_type=entry.get("target_type", ""),
    )


def _unique_anchor(base: str, counters: dict[str, int]) -> str:
    """Return a collision-free anchor ID by appending a counter if needed.

    On first use of *base*, returns *base* and records count 1.
    On collision, appends ``-2``, ``-3``, etc.
    """
    count = counters.get(base, 0)
    counters[base] = count + 1
    if count == 0:
        return base
    return f"{base}-{count + 1}"


# ===================================================================
# JsonTraceReader Adapter (TR-06)
# ===================================================================


class JsonTraceReader:
    """Reads trace.json and builds a TraceabilityContext.

    Satisfies the TraceReader Protocol via structural subtyping.
    Constructor accepts a default trace path; ``read()`` accepts
    an optional override.

    Usage::

        reader = JsonTraceReader(".research-to-dev/pipeline/trace.json")
        ctx = reader.read()  # uses default path
        ctx = reader.read("/custom/path/trace.json")  # overrides
    """

    def __init__(
        self, trace_path: str = ".research-to-dev/pipeline/trace.json"
    ) -> None:
        self._default_trace_path = trace_path

    def read(self, trace_path: str = "") -> TraceabilityContext | None:
        """Read trace data and build a TraceabilityContext.

        Args:
            trace_path: Override path to trace.json. If empty, uses the
                path provided to ``__init__``.

        Returns:
            Populated TraceabilityContext, or None if trace data is
            unavailable or unreadable.
        """
        import json
        import logging
        from pathlib import Path

        _logger = logging.getLogger(__name__)

        path_str = trace_path or self._default_trace_path
        file_path = Path(path_str)

        if not file_path.exists():
            _logger.warning(
                "Trace file not found at %s — no traceability data.", path_str
            )
            return None

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            _logger.warning(
                "Failed to parse trace.json at %s: %s — no traceability data.",
                path_str, exc,
            )
            return None

        # Extract hypotheses (top-level "hypotheses" key)
        hypothesis_dicts: list[dict] = data.get("hypotheses", [])
        if not isinstance(hypothesis_dicts, list):
            hypothesis_dicts = []

        # Extract correlation data (top-level "correlation" key)
        correlation_dict: dict | None = data.get("correlation")
        if not isinstance(correlation_dict, dict):
            correlation_dict = None

        # No data at all → None
        if not hypothesis_dicts and correlation_dict is None:
            return None

        return build_traceability(hypothesis_dicts, correlation_dict)
