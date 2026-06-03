"""Paper Profile — section and claim extraction for ranked papers.

Public API
----------
- **Pipeline**: ``ProfilingPipeline`` — orchestrates per-paper profiling
- **Data**: ``Section``, ``Claim``, ``PaperProfile``, ``ProfilingResult``, ``ProfilingError``
- **Protocols**: ``SectionExtractor`` — pluggable dependency for extraction
- **Adapters**: ``OpenAISectionExtractor`` — OpenAI implementation
- **Config**: ``ProfileConfig`` — constructor-injected configuration
"""

from research_to_dev.paper_profile.adapters import OpenAISectionExtractor
from research_to_dev.paper_profile.pipeline import ProfilingPipeline
from research_to_dev.paper_profile.protocols import SectionExtractor
from research_to_dev.paper_profile.types import (
    Claim,
    PaperProfile,
    ProfilingError,
    ProfilingResult,
    Section,
)
from research_to_dev.shared.config import ProfileConfig

__all__ = [
    "Claim",
    "OpenAISectionExtractor",
    "PaperProfile",
    "ProfileConfig",
    "ProfilingError",
    "ProfilingPipeline",
    "ProfilingResult",
    "Section",
    "SectionExtractor",
]
