"""Experiment setup module — config init, metric registry, trace reading,
program.md generation, git operations, and experiment orchestration.

Public API follows the Screaming Architecture convention: every file
exports the types and functions its name suggests.

Import submodules directly:
    from research_to_dev.experiment.metrics import MetricRegistry
    from research_to_dev.experiment.setup import ExperimentSetup
    ...
"""

__all__ = [
    "ExperimentSetup",
    "GitOperations",
    "MetricRegistry",
    "ProgramWriter",
    "extract_hypotheses",
    "load_config_yaml",
    "read_trace",
]
