# research-to-dev

Connect academic papers to your codebase — from literature search to experiment reports.

research-to-dev is a 4-phase pipeline that searches academic papers, correlates their claims with your code, generates improvement hypotheses, runs automated experiments, and produces actionable reports with full traceability back to the original research.

## Install

```bash
pip install -e .
```

Set required environment variables:

```bash
export OPENAI_API_KEY=sk-...
# Optional: enable Tavily as a third retrieval source
export TAVILY_API_KEY=tvil-...
# Optional: increase Semantic Scholar rate limits
export S2_API_KEY=...
```

## Quick Start

```bash
# Phase 1: Run the full research pipeline
research-to-dev analyze --query "transformer attention mechanisms" --codebase ./src

# Phase 2: Review hypotheses (human checkpoint) — check trace.json

# Phase 3: Set up and run an experiment
research-to-dev experiment setup \
  --hypothesis-id <HASH> \
  --time-budget 30m \
  --max-iterations 5 \
  --run-command "python train.py" \
  --baseline "val_accuracy=0.72" \
  --coding-agent-model "opencode-go/deepseek-v4-pro"

research-to-dev experiment run <HASH>

# Phase 4: Compile reports
research-to-dev report
research-to-dev report --with-insights
research-to-dev report --trace .research-to-dev/pipeline/trace.json
```

## The Pipeline

```
Phase 1: Research Collection          Phase 2: Hypothesis Generation
=================================    =================================
research-to-dev analyze              (human checkpoint — review trace.json)
  1. Search papers (arxiv, S2, Tavily)
  2. Extract content                     Generates hypotheses with narrative,
  3. Rank by relevance                   structure, and success metrics
  4. Profile papers (sections, claims)   based on paper↔code correlations
  5. Analyze codebase
  6. Correlate claims↔code             ───────── RED CHECKPOINT ─────────
  7. Generate hypotheses

                                       Phase 3: Experiment Execution
                                       =================================
                                       research-to-dev experiment setup
                                       research-to-dev experiment run
                                         Agentic loop: modify code → execute
                                         → extract metrics → keep/discard

                                       ───────── GREEN REVIEW ─────────

Phase 4: Report & Decision
=================================
research-to-dev report
  Compile results, generate LLM insights,
  add traceability, produce terminal + MD + JSON
```

## CLI Reference

### `research-to-dev analyze`

Run the full 7-step research pipeline: search → extract → rank → profile → codebase → correlate → hypothesize.

```
research-to-dev analyze [OPTIONS]
```

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--query` | `-q` | Research question to explore | *(required)* |
| `--codebase` | `-c` | Path to the codebase directory | `.` |
| `--output` | `-o` | JSON output file path | `.research-to-dev/results.json` |

### `research-to-dev config init`

Initialize a project config with documented built-in metric patterns. Idempotent — fails if the file already exists.

```
research-to-dev config init [OPTIONS]
```

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--path` | `-p` | Path to write the config file | `.research-to-dev/config.yaml` |

### `research-to-dev experiment setup`

Set up an isolated experiment for a hypothesis. Creates a git branch `experiment/<id>`, generates `program.md`, and validates the target metric.

```
research-to-dev experiment setup [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--hypothesis-id` | SHA-256 ID of the hypothesis | *(required)* |
| `--time-budget` | Duration (e.g. `30m`, `2h`, `1h30m`) | *(required)* |
| `--max-iterations` | Maximum improvement iterations (>= 1) | *(required)* |
| `--run-command` | Shell command to execute (e.g. `python train.py`) | *(required)* |
| `--baseline` | Baseline metric as `key=value` (e.g. `val_accuracy=0.72`) | *(required)* |
| `--coding-agent-model` | Model identifier for the coding agent | *(required)* |
| `--direction` | `maximize` or `minimize` | Inferred from success_criteria |
| `--trace` | `-t` | Path to pipeline trace JSON | `.research-to-dev/pipeline/trace.json` |
| `--config` | Path to config.yaml for custom metric patterns | `.research-to-dev/config.yaml` |

### `research-to-dev experiment run`

Run the experiment iteration loop for a hypothesis. All configuration comes from `program.md` — no CLI overrides.

```
research-to-dev experiment run <HYPOTHESIS_ID>
```

| Argument | Description |
|----------|-------------|
| `HYPOTHESIS_ID` | SHA-256 ID of the hypothesis (positional) |

### `research-to-dev report`

Compile experiment results into markdown and JSON reports.

```
research-to-dev report [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--hypothesis` | Compile report for a specific hypothesis ID only | All hypotheses |
| `--with-insights` | Include LLM-generated cross-hypothesis insights | `False` |
| `--trace` | `-t` | Path to pipeline trace.json for traceability links | `.research-to-dev/pipeline/trace.json` |

Reports are written to `.research-to-dev/reports/` with timestamped filenames like `report-20260612-143000.md` and `report-20260612-143000.json`.

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM calls (ranking, profiling, correlation, hypothesis, insights) |
| `TAVILY_API_KEY` | No | Enables Tavily as an additional paper retrieval source |
| `S2_API_KEY` | No | Increases Semantic Scholar API rate limits |

### config.yaml

Custom metric patterns for experiment metric extraction. Generate with:

```bash
research-to-dev config init
```

Produces `.research-to-dev/config.yaml`:

```yaml
# Research-to-Dev configuration file.
# Generated by `research-to-dev config init`.
#
# Regex patterns use Python's `re` module syntax.
# Example: r"accuracy[:\s=]*([\d.]+)"

metrics:
  # val_loss: "val_loss[:\s=]*([\d.]+)"  # built-in default
  # accuracy: "accuracy[:\s=]*([\d.]+)"  # built-in default
  # loss: "loss[:\s=]*([\d.]+)"  # built-in default
  # f1: "f1[:\s=]*([\d.]+)"  # built-in default
  # bleu: "bleu[:\s=]*([\d.]+)"  # built-in default
  # perplexity: "perplexity[:\s=]*([\d.]+)"  # built-in default

  # Custom metric examples:
  # custom_auc: "auc[:\s=]*([\d.]+)"
```

Built-in metrics are always available. Config entries with the same name override built-in patterns. Custom entries add new metrics.

## Output Formats

### Directory Structure

```
.research-to-dev/
├── config.yaml                  # Metric patterns and project settings
├── results.json                 # Analyze output
├── experiments/<hypothesis-id>/
│   ├── program.md               # Experiment objective and configuration
│   ├── results.tsv              # Iteration log (keep/discard history)
│   └── logs/                    # Training stdout/stderr per iteration
├── reports/
│   ├── report-YYYYMMDD-HHMMSS.md
│   └── report-YYYYMMDD-HHMMSS.json
└── pipeline/
    ├── trace.json               # Full pipeline trace with all 7 steps
    └── correlation-map.json     # Paper↔code correlation data
```

### Terminal Summary

After `analyze`:

```
════════════════════════════════════════
 PIPELINE RESULTS
════════════════════════════════════════
Query: "transformer attention mechanisms" | Papers: 8 ranked | Codebase: 12 modules | Correlations: 15

 Top Hypotheses:
  #  Score  Hypothesis
  1   8.2  Integrate multi-head attention pruning...
  2   7.5  Replace feed-forward layers with...
  3   6.8  Apply layer-wise learning rate...
  4   5.9  Use gradient checkpointing for...
  5   5.1  Implement sparse attention over...

 Output:   results.json
 Time:     45.2s
════════════════════════════════════════
```

After `experiment run`:

```
════════════════════════════════════════
 EXPERIMENT COMPLETED
════════════════════════════════════════
  Iterations:     5
  Kept:           3 (improvements committed)
  Discarded:      2 (failed/timeout/worse)
  Final baseline: val_accuracy=0.781
════════════════════════════════════════
```

After `report`:

```
════════════════════════════════════════
 REPORT SUMMARY
════════════════════════════════════════

 Top Hypotheses:
  #     Delta   Title                                              Status
  1   +8.3%   a3f2c1...                                         improved
  2   +2.1%   b7d9e4...                                         improved
  3   -1.4%   c5a8f0...                                         worsened

 Top Recommendations:
  1. Prioritize attention pruning across all attention-ba...
  2. Combine gradient checkpointing with mixed-precision ...

 Verdict: 2/3 hypotheses improved vs baseline
 Output:   .research-to-dev/reports/report-20260612-143000.md
 Output:   .research-to-dev/reports/report-20260612-143000.json
 Time:     2.1s
════════════════════════════════════════
```

### Markdown Report

```markdown
# Experiment Results — 2026-06-12T14:30:00+00:00

## Hypothesis: a3f2c1e...

- **Status**: improved
- **Iterations**: 5 total, 3 kept, 2 discarded
- **Best metric**: val_accuracy=0.781 (delta: +0.0610)
- **Baseline comparison**: improved (best improvement: +0.0610)

| # | Metric | Value | Delta | Status |
|---|--------|-------|-------|--------|
| 1 | val_accuracy | 0.741 | +0.0210 | success |
| 2 | val_accuracy | 0.765 | +0.0450 | success |
| 3 | val_accuracy | 0.72 | -0.0000 | success |
| 4 | val_accuracy | 0.781 | +0.0610 | success |
| 5 | val_accuracy | 0.71 | -0.0100 | success |

### Traceability
...
```

When `--with-insights` is passed, the report includes:

```markdown
---

## Cross-Hypothesis Insights

### Patterns

**Attention pruning improves convergence** [high]
Multi-head attention pruning consistently reduced training time
without accuracy loss across 3 hypotheses.

### Recommendations

- **[high]** Prioritize attention pruning for all transformer layers
  Reduces compute by ~30% with minimal accuracy tradeoff.
  Evidence: a3f2c1e..., b7d9e4...

### Evidence Correlations

**arXiv:2304.12345**
Multi-head attention can be pruned based on gradient magnitude.
Hypotheses: a3f2c1e..., b7d9e4... · Code: src/model/attention.py
```

When `--trace` is passed, each hypothesis section includes a Traceability subsection linking back to the original paper claims and code components, plus a Traceability Appendix at the end with full claim text and component signatures.

### JSON Report

```json
{
  "generated_at": "2026-06-12T14:30:00+00:00",
  "hypotheses": [
    {
      "hypothesis_id": "a3f2c1e...",
      "status": "improved",
      "iterations_total": 5,
      "iterations_kept": 3,
      "iterations_discarded": 2,
      "iterations_crashed": 0,
      "best_metric_value": 0.781,
      "baseline_comparison": {
        "best_improvement": 0.061,
        "overall_trend": "improved"
      },
      "iterations": [
        {
          "iteration": 1,
          "metric_name": "val_accuracy",
          "metric_value": 0.741,
          "baseline_value": 0.72,
          "delta": 0.021,
          "status": "success",
          "timestamp": "2026-06-12T14:31:00Z"
        }
      ]
    }
  ],
  "insights": {
    "patterns": [...],
    "recommendations": [...],
    "evidence_correlation": [...]
  },
  "traceability": {
    "hypothesis_origins": {...},
    "claim_anchors": {...},
    "component_anchors": {...}
  }
}
```

## License

MIT