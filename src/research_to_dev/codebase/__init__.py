"""Codebase Analysis — AST scanning + LLM component descriptions.

Public API
----------
- **Pipeline**: ``CodebaseAnalysisPipeline`` — orchestrates scan → describe → assemble
- **Data**: ``Component``, ``ModuleSummary``, ``CodebaseContext``, ``CodebaseError``
- **Protocols**: ``CodebaseDescriber`` — pluggable dependency for descriptions
- **Adapters**: ``OpenAIDescriber`` — OpenAI implementation
- **Scanner**: ``scan_project``, ``scan_file`` — AST-based Python code scanning
- **Config**: ``CodebaseConfig`` — constructor-injected configuration
"""

from research_to_dev.codebase.adapters import OpenAIDescriber
from research_to_dev.codebase.pipeline import CodebaseAnalysisPipeline
from research_to_dev.codebase.protocols import CodebaseDescriber
from research_to_dev.codebase.scanner import scan_file, scan_project
from research_to_dev.codebase.types import (
    CodebaseContext,
    CodebaseError,
    Component,
    ModuleSummary,
)
from research_to_dev.shared.config import CodebaseConfig

__all__ = [
    "CodebaseAnalysisPipeline",
    "CodebaseConfig",
    "CodebaseContext",
    "CodebaseDescriber",
    "CodebaseError",
    "Component",
    "ModuleSummary",
    "OpenAIDescriber",
    "scan_file",
    "scan_project",
]
