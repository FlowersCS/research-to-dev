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

__all__ = [
    "BaselineComparison",
    "CompiledReport",
    "HypothesisSummary",
    "IterationResult",
    "ResultsReader",
    "TsvResultsReader",
    "compile_all",
    "compile_hypothesis",
    "format_json",
    "format_markdown",
    "write_reports",
]
