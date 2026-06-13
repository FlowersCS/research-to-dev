"""Report compilation module — reads per-hypothesis experiment results
and produces structured markdown + JSON reports.

Public API follows the Screaming Architecture convention:
    from research_to_dev.report import (
        ResultsReader,
        TsvResultsReader,
        compile_hypothesis,
        compile_all,
        format_markdown,
        format_json,
    )
"""

from research_to_dev.report.compilation import (
    BaselineComparison,
    CompiledReport,
    HypothesisSummary,
    IterationResult,
    ResultsReader,
    TsvResultsReader,
    compile_all,
    compile_hypothesis,
    format_json,
    format_markdown,
    write_reports,
)
from research_to_dev.report.insights import (
    EvidenceCorrelation,
    InsightsGenerator,
    InsightsReport,
    OpenAIInsightsGenerator,
    PatternInsight,
    ProgramContext,
    Recommendation,
)
from research_to_dev.report.traceability import (
    CorrelationTrace,
    HypothesisOrigin,
    JsonTraceReader,
    TraceReader,
    TraceabilityContext,
    build_traceability,
)

__all__ = [
    "BaselineComparison",
    "CompiledReport",
    "CorrelationTrace",
    "EvidenceCorrelation",
    "HypothesisOrigin",
    "HypothesisSummary",
    "InsightsGenerator",
    "InsightsReport",
    "IterationResult",
    "JsonTraceReader",
    "OpenAIInsightsGenerator",
    "PatternInsight",
    "ProgramContext",
    "Recommendation",
    "ResultsReader",
    "TraceReader",
    "TraceabilityContext",
    "TsvResultsReader",
    "build_traceability",
    "compile_all",
    "compile_hypothesis",
    "format_json",
    "format_markdown",
    "write_reports",
]
