"""Shared configuration dataclasses for the research-to-dev pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProfileConfig:
    """Configuration for the Paper Profiling pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    model: str = "gpt-4o-mini"
    """OpenAI chat model used for section and claim extraction."""

    max_claims_per_paper: int = 20
    """Maximum number of claims to keep per paper (caps output size)."""


@dataclass
class RankingConfig:
    """Configuration for the Ranking Funnel pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    min_year: int = field(default_factory=lambda: datetime.now().year - 5)
    """Hard filter threshold — papers dated before this year are excluded."""

    semantic_top_k: int = 20
    """Number of top papers to select after semantic ranking (Stage 2)."""

    llm_min_papers: int = 5
    """Minimum number of papers the LLM should return (Stage 3)."""

    llm_max_papers: int = 10
    """Maximum number of papers the LLM should return (Stage 3)."""

    embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model used for semantic ranking."""

    llm_model: str = "gpt-4o-mini"
    """OpenAI chat model used for relevance judgment."""


@dataclass
class CorrelationMapConfig:
    """Configuration for the CorrelationMap pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    similarity_threshold: float = 0.4
    """Cosine similarity cutoff — pairs below this score are discarded (CM-CFG-01)."""

    embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model used for similarity computation (CM-CFG-01)."""

    llm_model: str = "gpt-4o-mini"
    """OpenAI chat model used for correlation classification (CM-CFG-01)."""


@dataclass
class HypothesisConfig:
    """Configuration for the Hypothesis Generator pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    llm_model: str = "gpt-4o-mini"
    """OpenAI chat model used for hypothesis generation and reflection."""

    embedding_model: str = "text-embedding-3-small"
    """OpenAI embedding model used for deduplication."""

    max_reflection_rounds: int = 3
    """Maximum number of reflection rounds before forced stop (AD-09)."""

    dedup_threshold: float = 0.85
    """Cosine similarity threshold for deduplication (AD-06, AD-09)."""

    convergence_score_delta: float = 0.5
    """Maximum average composite score change for convergence detection (AD-05, AD-09)."""

    max_hypotheses_per_paper: int = 5
    """Maximum hypotheses to keep per paper-cluster after generation (AD-09)."""

    max_total_hypotheses: int = 30
    """Hard cap on total hypotheses in the merged result (AD-09)."""

    relevance_gate: int = 7
    """Minimum relevance score (1-10) — hypotheses below this are filtered (AD-09)."""


@dataclass
class MetricConfig:
    """Configuration for a single metric pattern in the registry.

    Constructor-injectable so tests can supply arbitrary metric configs
    without touching ``.research-to-dev/config.yaml`` or global state.
    """

    name: str
    """Metric identifier (e.g. ``val_loss``, ``accuracy``)."""

    pattern: str
    r"""Regex pattern for extraction (e.g. ``val_loss[:\s=]*([\d.]+)``)."""


@dataclass
class ExperimentConfig:
    """Configuration for experiment setup.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching CLI flags or filesystem state.
    """

    time_budget: str = ""
    """Duration string e.g. ``"30m"``, ``"2h"``, ``"1h30m"`` — validated via regex at runtime."""

    max_iterations: int = 0
    """Positive integer — validated at runtime. Must be >= 1."""

    trace_path: str = ".research-to-dev/pipeline/trace.json"
    """Path to the PipelineTrace JSON file (default from D2)."""

    config_path: str = ".research-to-dev/config.yaml"
    """Path to the optional per-project metric config file."""

    hypothesis_id: str = ""
    """SHA-256 ID of the hypothesis to set up the experiment for."""


@dataclass
class RunConfig:
    """Internal representation of experiment run configuration.

    Mirrors what ``program_reader`` produces from ``program.md`` frontmatter.
    This is for internal representation, NOT for CLI flags (D11 — single
    source of truth is program.md).

    All values are constructor-injectable so tests can supply arbitrary
    config without touching the filesystem.
    """

    agent_model: str
    """Model identifier for the coding agent (e.g. ``"opencode-go/deepseek-v4-pro"``)."""

    time_budget_seconds: int
    """Total time budget in seconds (parsed from ``time_budget`` string)."""

    max_iterations: int
    """Maximum number of agent-code-execute iterations (must be >= 1)."""

    baseline: dict[str, float]
    """Baseline metric values (e.g. ``{"val_accuracy": 0.72}``)."""

    run_command: str
    """Shell command to execute (e.g. ``"python train.py"``)."""

    direction: str = "maximize"
    """Direction for improvement comparison: ``"maximize"`` or ``"minimize"``."""


@dataclass
class ReportConfig:
    """Configuration for the Results Compilation pipeline (D11).

    All values are constructor-injectable so tests can supply arbitrary
    config without touching the filesystem.
    """

    experiments_dir: str = ".research-to-dev/experiments"
    """Directory containing per-hypothesis experiment subdirectories."""

    reports_dir: str = ".research-to-dev/reports"
    """Directory where compiled ``.md`` and ``.json`` reports are written."""

    hypothesis_filter: str | None = None
    """Optional hypothesis ID to filter compilation to a single hypothesis."""

    include_insights: bool = False
    """Whether to include LLM-generated insights (no-op placeholder for now)."""


@dataclass
class CodebaseConfig:
    """Configuration for the Codebase Analysis pipeline.

    All values are constructor-injectable so tests can supply arbitrary
    config without touching ``.env`` or global state.
    """

    include_private: bool = False
    """Whether to include ``_``-prefixed functions and classes."""

    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "test_*.py",
            "*_test.py",
            "setup.py",
            "conftest.py",
        ]
    )
    """Glob patterns for files to exclude from scanning."""

    include_patterns: list[str] = field(default_factory=lambda: ["*.py"])
    """Glob patterns for files to include in scanning."""

    max_components_per_module: int = 200
    """Maximum number of components to extract from a single module."""

    model: str = "gpt-4o-mini"
    """OpenAI chat model used for component description and module summaries."""
