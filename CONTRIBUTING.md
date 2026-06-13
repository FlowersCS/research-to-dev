# Contributing to research-to-dev

## Architecture

research-to-dev follows **Screaming Architecture** (module names reveal domain intent) combined with **Hexagonal Architecture** (domain logic isolated behind Protocols, adapters provide external dependencies).

```
src/research_to_dev/
├── cli/                    # CLI entry point (Typer)
│   ├── main.py             # Commands: analyze, config, experiment, report
│   └── orchestrator.py     # PipelineOrchestrator — wires all 7 steps
├── retriever/              # Phase 1: Search papers (arxiv, S2, Tavily)
├── extraction/             # Phase 1: Extract content from papers
├── ranking/                # Phase 1: Rank papers by relevance
├── paper_profile/          # Phase 1: Profile papers (sections, claims)
├── codebase/               # Phase 1: Analyze codebase structure
├── correlation/            # Phase 1: Correlate claims ↔ code
├── hypothesis/             # Phase 2: Generate improvement hypotheses
├── experiment/             # Phase 3: Execute experiments
│   ├── agent.py            # OpenCodeAdapter (coding agent protocol)
│   ├── executor.py         # CodeExecutor (runs shell commands)
│   ├── git.py              # GitOperations (branch, commit, rollback)
│   ├── metrics.py          # MetricRegistry + keep/discard logic
│   ├── program.py           # ProgramWriter (generates program.md)
│   ├── program_reader.py    # parse_program_md (reads program.md)
│   ├── results.py           # ResultsWriter (TSV logging)
│   ├── runner.py            # ExperimentRunner (iteration loop)
│   ├── setup.py             # ExperimentSetup (branch + program.md)
│   └── trace.py             # extract_hypotheses, read_trace
├── report/                 # Phase 4: Compile results
│   ├── compilation.py       # format_markdown, format_json, compile_all
│   ├── insights.py          # LLM cross-hypothesis insights
│   └── traceability.py      # Paper→claim→code traceability chain
└── shared/
    └── config.py           # Config dataclasses (constructor injection)
```

## Design Patterns

### Protocol Pattern

All interfaces are `typing.Protocol` classes. Adapters are concrete implementations injected at construction time. This keeps domain logic decoupled from external services.

Examples:
- `ResultsReader` Protocol → `TsvResultsReader` adapter (reads `results.tsv` files)
- `InsightsGenerator` Protocol → `OpenAIInsightsGenerator` adapter (LLM calls)
- `TraceReader` Protocol → `JsonTraceReader` adapter (reads `trace.json`)

### Constructor Injection

All dependencies are passed via `__init__`, never imported as globals. This enables test doubles without monkeypatching and makes dependency boundaries explicit.

### Error Resilience Tiers

Four-tier error handling strategy:

| Tier | Strategy | Use Case | Example |
|------|----------|----------|---------|
| T1 | Fail-fast | Invalid input that can't be recovered | Missing `OPENAI_API_KEY` |
| T2 | Degrade-gracefully | Partial failures where some output is still valuable | Paper retrieval returns partial results |
| T3 | Best-effort | Non-critical steps that can be skipped | Insights generation failure doesn't abort reporting |
| T4 | Skip-and-continue | Optional data that should not block the pipeline | Malformed correlation entries are skipped |

## Module Guide

### `cli/`

**Purpose**: CLI entry points and pipeline orchestration.

- `main.py` — Typer app with commands: `analyze`, `config init`, `experiment setup`, `experiment run`, `report`
- `orchestrator.py` — `PipelineOrchestrator` wires all 7 steps and manages `PipelineTrace`

### `retriever/`

**Purpose**: Phase 1, Step 1 — Search academic papers from arxiv, Semantic Scholar, and optionally Tavily.

### `extraction/`

**Purpose**: Phase 1, Step 2 — Extract and clean content from paper HTML/PDF.

### `ranking/`

**Purpose**: Phase 1, Step 3 — Three-stage ranking funnel (keyword → semantic → LLM judgment) with `RankingConfig`.

### `paper_profile/`

**Purpose**: Phase 1, Step 4 — Extract sections and claims from papers using LLM, producing structured paper profiles with `ProfileConfig`.

### `codebase/`

**Purpose**: Phase 1, Step 5 — Analyze source code structure: extract components (functions, classes) and module summaries with `CodebaseConfig`.

### `correlation/`

**Purpose**: Phase 1, Step 6 — Map paper claims to code components via embedding similarity and LLM classification with `CorrelationMapConfig`.

### `hypothesis/`

**Purpose**: Phase 2 — Generate improvement hypotheses from correlations, with self-reflection and deduplication, governed by `HypothesisConfig`.

### `experiment/`

**Purpose**: Phase 3 — Set up and run experiments in isolated git branches. Key submodules:

- `setup.py` — Creates branch, writes `program.md`
- `runner.py` — Iteration loop: agent modifies code → executor runs command → metrics extracted → keep/discard decision
- `agent.py` — `OpenCodeAdapter` sends modification instructions to a coding agent
- `metrics.py` — `MetricRegistry` extracts metrics from command output using regex patterns
- `program_reader.py` — Parses `program.md` frontmatter into `ProgramSpec`

### `report/`

**Purpose**: Phase 4 — Compile experiment results.

- `compilation.py` — `compile_all`, `compile_hypothesis`, `format_markdown`, `format_json`, `write_reports`, `TsvResultsReader`
- `insights.py` — `OpenAIInsightsGenerator` produces cross-hypothesis patterns, recommendations, and evidence correlations via LLM
- `traceability.py` — Builds `TraceabilityContext` linking hypotheses back to paper claims and code components. `JsonTraceReader` adapter reads pipeline `trace.json`

Key types:
- `CompiledReport` — top-level report with hypotheses, optional insights, optional traceability
- `HypothesisSummary` — per-hypothesis aggregation (status, kept/discarded counts, baseline comparison)
- `IterationResult` — single row from `results.tsv`
- `BaselineComparison` — improvement trend vs baseline
- `InsightsReport` — patterns, recommendations, evidence correlations
- `TraceabilityContext` — hypothesis origins, claim anchors, component anchors

### `shared/config.py`

**Purpose**: All pipeline configuration dataclasses. See Configuration Internals below.

## Configuration Internals

All config dataclasses live in `shared/config.py` and are constructor-injected. Defaults are sensible for typical use; tests override them as needed.

### `RankingConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_year` | `int` | Current year - 5 | Hard filter: papers before this year are excluded |
| `semantic_top_k` | `int` | `20` | Number of top papers after semantic ranking |
| `llm_min_papers` | `int` | `5` | Minimum papers the LLM should return |
| `llm_max_papers` | `int` | `10` | Maximum papers the LLM should return |
| `embedding_model` | `str` | `"text-embedding-3-small"` | OpenAI embedding model for semantic ranking |
| `llm_model` | `str` | `"gpt-4o-mini"` | OpenAI chat model for relevance judgment |

### `ProfileConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | OpenAI chat model for section/claim extraction |
| `max_claims_per_paper` | `int` | `20` | Max claims to keep per paper |

### `CorrelationMapConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `similarity_threshold` | `float` | `0.4` | Cosine similarity cutoff; pairs below are discarded |
| `embedding_model` | `str` | `"text-embedding-3-small"` | OpenAI embedding model for similarity |
| `llm_model` | `str` | `"gpt-4o-mini"` | OpenAI chat model for correlation classification |

### `HypothesisConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_model` | `str` | `"gpt-4o-mini"` | OpenAI chat model for hypothesis generation/reflection |
| `embedding_model` | `str` | `"text-embedding-3-small"` | OpenAI embedding model for deduplication |
| `max_reflection_rounds` | `int` | `3` | Maximum reflection rounds before forced stop |
| `dedup_threshold` | `float` | `0.85` | Cosine similarity threshold for deduplication |
| `convergence_score_delta` | `float` | `0.5` | Max average composite score change for convergence |
| `max_hypotheses_per_paper` | `int` | `5` | Max hypotheses per paper-cluster |
| `max_total_hypotheses` | `int` | `30` | Hard cap on total hypotheses |
| `relevance_gate` | `int` | `7` | Minimum relevance score (1-10); below = filtered |

### `CodebaseConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `include_private` | `bool` | `False` | Whether to include `_`-prefixed functions/classes |
| `exclude_patterns` | `list[str]` | `["test_*.py", "*_test.py", "setup.py", "conftest.py"]` | Glob patterns for files to exclude |
| `include_patterns` | `list[str]` | `["*.py"]` | Glob patterns for files to include |
| `max_components_per_module` | `int` | `200` | Max components to extract per module |
| `model` | `str` | `"gpt-4o-mini"` | OpenAI chat model for descriptions/summaries |

### `ExperimentConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `time_budget` | `str` | `""` | Duration string (e.g. `"30m"`, `"2h"`, `"1h30m"`) |
| `max_iterations` | `int` | `0` | Positive integer, validated at runtime (must be >= 1) |
| `trace_path` | `str` | `".research-to-dev/pipeline/trace.json"` | Path to PipelineTrace JSON |
| `config_path` | `str` | `".research-to-dev/config.yaml"` | Path to optional per-project metric config |
| `hypothesis_id` | `str` | `""` | SHA-256 ID of the target hypothesis |

### `RunConfig`

Internal representation of experiment run config, parsed from `program.md` frontmatter. Not exposed as CLI flags (D11 — single source of truth is `program.md`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_model` | `str` | *(required)* | Model identifier for the coding agent |
| `time_budget_seconds` | `int` | *(required)* | Total time budget in seconds |
| `max_iterations` | `int` | *(required)* | Max agent-code-execute iterations (>= 1) |
| `baseline` | `dict[str, float]` | *(required)* | Baseline metric values (e.g. `{"val_accuracy": 0.72}`) |
| `run_command` | `str` | *(required)* | Shell command to execute |
| `direction` | `str` | `"maximize"` | `"maximize"` or `"minimize"` |

### `ReportConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiments_dir` | `str` | `".research-to-dev/experiments"` | Directory with per-hypothesis subdirectories |
| `reports_dir` | `str` | `".research-to-dev/reports"` | Output directory for compiled reports |
| `hypothesis_filter` | `str \| None` | `None` | Filter compilation to a single hypothesis |
| `include_insights` | `bool` | `False` | Whether to generate LLM cross-hypothesis insights |
| `insights_model` | `str` | `"gpt-4o-mini"` | OpenAI chat model for insights generation |
| `trace_path` | `str` | `""` | Path to `trace.json` for traceability; empty = disabled |

### `MetricConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Metric identifier (e.g. `val_loss`, `accuracy`) |
| `pattern` | `str` | *(required)* | Regex pattern for extraction |

## Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Or with uv
uv sync --group dev

# Run tests
pytest

# Run a specific test
pytest tests/ranking/test_pipeline.py -v

# Run tests for a single module
pytest tests/correlation/ -v
```

Dev dependencies: `pytest>=8.0.0`, `pytest-asyncio>=0.24.0`, `respx>=0.21.0`.

## Testing

The test suite has 62 test files across all modules. Test structure mirrors `src/`:

```
tests/
├── cli/           # CLI command tests
├── codebase/      # Codebase analysis tests
├── correlation/   # Correlation pipeline tests
├── experiment/    # Experiment setup/run tests
├── extraction/    # Paper extraction tests
├── hypothesis/    # Hypothesis generation tests
├── paper_profile/ # Paper profiling tests
├── ranking/       # Ranking funnel tests
├── report/        # Report compilation tests
├── retriever/     # Paper retrieval tests
└── shared/        # Shared config tests
```

Each module directory contains:
- `conftest.py` — shared fixtures and test doubles
- `test_pipeline.py` — integration tests for the module pipeline
- `test_adapters.py` — adapter/Protocol implementation tests
- `test_types.py` — dataclass and type validation tests

## SDD Workflow

This project uses Spec-Driven Development (SDD) for substantial changes. The workflow:

1. **Explore** — Investigate the codebase and requirements before committing
2. **Propose** — Write a change proposal with intent, scope, and approach
3. **Specify** — Write requirements and scenarios (delta specs)
4. **Design** — Create a technical design document
5. **Tasks** — Break down into implementation task checklist
6. **Apply** — Implement tasks, checking off items
7. **Verify** — Validate implementation against specs
8. **Archive** — Close the change and persist final state

All SDD artifacts live in persistent memory (Engram) or `openspec/` files.

## Commit Convention

Use Conventional Commits with CDE (Context-Description-Effect) format:

```
type(scope): short description in imperative mood

Why:
- Context and motivation for the change

What:
- List of changes made
```

Rules:
- **type**: `feat`, `fix`, `refactor`, `style`, `docs`, `test`, `chore`
- **scope**: optional but recommended — the module/area affected
- **Description**: imperative mood, short, no period
- Separate commits by logical purpose, not by file

## Project Status

All 4 pipeline phases are complete and merged:

| Phase | Status | Modules |
|-------|--------|---------|
| Phase 1: Research Collection | Complete | retriever, extraction, ranking, paper_profile, codebase, correlation |
| Phase 2: Hypothesis Generation | Complete | hypothesis |
| Phase 3: Experiment Execution | Complete | experiment (setup, runner, agent, metrics, git, results) |
| Phase 4: Report & Traceability | Complete | report (compilation, insights, traceability) |