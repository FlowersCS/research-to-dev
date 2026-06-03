"""Codebase Analysis Pipeline — orchestrates scan → describe → assemble.

Orchestrates the full codebase analysis pipeline: scans via
``scanner.py``, generates LLM descriptions per module via the
plugged-in ``CodebaseDescriber`` Protocol, and assembles a
``CodebaseContext``. Follows the same constructor-injection pattern
as ``RankingPipeline`` and ``ProfilingPipeline``.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

from research_to_dev.codebase.protocols import CodebaseDescriber
from research_to_dev.codebase.scanner import scan_project
from research_to_dev.codebase.types import (
    CodebaseContext,
    Component,
    ModuleSummary,
)
from research_to_dev.shared.config import CodebaseConfig

_logger = logging.getLogger(__name__)


class CodebaseAnalysisPipeline:
    """Pipeline that orchestrates codebase scanning, description, and assembly.

    Constructor-injected config and ``CodebaseDescriber`` Protocol
    implementation keep the pipeline testable without real API calls
    (CA-10). The pipeline never raises (CA-09) — errors produce
    degraded output and the pipeline continues.
    """

    def __init__(
        self,
        describer: CodebaseDescriber,
        *,
        config: CodebaseConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._describer = describer
        self._config = config or CodebaseConfig()
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, project_path: str) -> CodebaseContext:
        """Run the full codebase analysis pipeline.

        1. Scan the project tree for ``.py`` files and extract raw components
        2. Group components by module and read source for each
        3. Per module: call the describer to get descriptions and a summary
        4. Assemble the final ``CodebaseContext``

        Args:
            project_path: Absolute or relative path to the project root.

        Returns:
            ``CodebaseContext`` with all components and module summaries.
            On any failure returns a context with empty components
            and modules — the pipeline never raises (CA-09).
        """
        project_name = os.path.basename(os.path.abspath(project_path))

        # Step 1: Scan
        try:
            raw_components, files_scanned = scan_project(
                project_path, self._config
            )
        except Exception as exc:
            self._logger.warning("Scan failed: %s", exc)
            return CodebaseContext(
                project_name=project_name,
                project_path=project_path,
                language="python",
            )

        total_components_found = len(raw_components)

        if not raw_components:
            return CodebaseContext(
                project_name=project_name,
                project_path=project_path,
                language="python",
                total_files_scanned=files_scanned,
                total_components_found=0,
            )

        # Step 2: Group components by module
        modules: dict[str, list[dict]] = defaultdict(list)
        file_paths: dict[str, str] = {}
        for comp in raw_components:
            module_name = comp["module_name"]
            modules[module_name].append(comp)
            if module_name not in file_paths:
                file_paths[module_name] = comp["file_path"]

        # Step 3: Describe each module (sequential, CA-05 batched per module)
        all_components: list[Component] = []
        all_summaries: list[ModuleSummary] = []

        for module_name, module_components in modules.items():
            file_path = file_paths[module_name]

            # Read the source for this module
            try:
                with open(file_path, encoding="utf-8") as fh:
                    source = fh.read()
            except Exception as exc:
                self._logger.warning(
                    "Failed to read source for module '%s': %s",
                    module_name,
                    exc,
                )
                source = ""

            # Call describer — errors produce degraded output (CA-07)
            try:
                result = await self._describer.describe(
                    module_path=module_name,
                    source=source,
                    components=module_components,
                )
            except Exception as exc:
                self._logger.warning(
                    "Describer failed for module '%s': %s",
                    module_name,
                    exc,
                )
                result = {
                    "descriptions": [],
                    "module_summary": "",
                    "key_responsibilities": [],
                }

            # Map descriptions back to components
            desc_map: dict[str, str] = {}
            for d in result.get("descriptions", []):
                if isinstance(d, dict):
                    name = d.get("name")
                    desc = d.get("description", "")
                    if isinstance(name, str) and isinstance(desc, str):
                        desc_map[name] = desc

            # Build Component domain objects
            for raw in module_components:
                name = raw.get("name", "")
                description = desc_map.get(name, raw.get("docstring") or "")
                all_components.append(
                    Component(
                        name=name,
                        kind=raw.get("kind", "function"),
                        file_path=raw.get("file_path", file_path),
                        line_start=raw.get("line_start", 0),
                        line_end=raw.get("line_end", 0),
                        signature=raw.get("signature", ""),
                        description=description,
                        module_name=raw.get("module_name", module_name),
                        docstring=raw.get("docstring"),
                    )
                )

            # Build ModuleSummary
            all_summaries.append(
                ModuleSummary(
                    module_name=module_name,
                    file_path=file_path,
                    summary=result.get("module_summary", ""),
                    key_responsibilities=result.get("key_responsibilities", []),
                )
            )

        # Step 4: Assemble
        return CodebaseContext(
            project_name=project_name,
            project_path=os.path.abspath(project_path),
            language="python",
            components=all_components,
            modules=all_summaries,
            total_files_scanned=files_scanned,
            total_components_found=total_components_found,
        )
