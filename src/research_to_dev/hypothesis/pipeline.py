"""Hypothesis generation pipeline — correlate → generate → reflect → assemble.

Monolithic pipeline with reflection loop as private methods (AD-01).
Follows CorrelationPipeline pattern: all stages are private methods
on the pipeline class. The pipeline never raises — every stage degrades
gracefully (HG-06, AD-07).
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

from research_to_dev.correlation.types import Correlation, CorrelationMap, ModuleCorrelation
from research_to_dev.extraction.types import GeneralContent
from research_to_dev.ranking.protocols import Embedder
from research_to_dev.shared.config import HypothesisConfig

from .protocols import HypothesisGenerator
from .types import Hypothesis, HypothesisError, HypothesisResult, HypothesisScores


class HypothesisPipeline:
    """Orchestrates correlation → generate → reflect → assemble.

    Reuses the shared ``Embedder`` Protocol from ``ranking.protocols``
    for deduplication embeddings (AD-06). The ``HypothesisGenerator`` is
    a new Protocol for LLM-based hypothesis creation and reflection.

    All dependencies are constructor-injected so the pipeline is fully
    testable with mock generators and embedders.
    """

    def __init__(
        self,
        *,
        generator: HypothesisGenerator,
        embedder: Embedder,
        config: HypothesisConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the pipeline with injected dependencies.

        Args:
            generator: ``HypothesisGenerator`` Protocol implementation
                (e.g. ``OpenAIHypothesisGenerator``).
            embedder: Shared ``Embedder`` Protocol implementation (e.g.
                ``OpenAIEmbedder`` from the ranking module).
            config: Configuration dataclass. Defaults to
                ``HypothesisConfig()``.
            logger: Logger instance. Defaults to
                ``logging.getLogger(__name__)``.
        """
        self._generator = generator
        self._embedder = embedder
        self._config = config or HypothesisConfig()
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        correlation_map: CorrelationMap,
        query: str,
        general_content: list[GeneralContent] | None = None,
    ) -> HypothesisResult:
        """Run the full hypothesis generation pipeline.

        Args:
            correlation_map: Validated correlations from the correlation
                module.
            query: Original user query for context.
            general_content: Optional list of ``GeneralContent`` items
                from web sources (tutorials, docs, blogs). Formatted as
                ``Implementation References`` section in LLM prompts.
                ``None`` or empty list works exactly as before.

        Returns:
            A ``HypothesisResult`` with scored, grounded hypotheses.
            Never raises — returns a degraded (possibly empty) result on
            failure (HG-06).
        """
        config = self._config
        errors: list[HypothesisError] = []

        # Format general content for prompt injection
        formatted_general = self._format_general_content(
            general_content or [],
        )

        # Merge both correlation types
        all_corrs: list[Correlation | ModuleCorrelation] = (
            list(correlation_map.correlations)
            + list(correlation_map.module_correlations)
        )

        if not all_corrs:
            return HypothesisResult(
                total_input_papers=0,
                total_correlations=0,
            )

        # --- Step 1: Build correlation index ---
        paper_corr_index, paper_titles, valid_ids = (
            self._build_correlation_index(all_corrs)
        )

        # --- Step 2: Generate initial hypotheses per paper-cluster ---
        all_hypotheses: list[Hypothesis] = []

        for paper_id, corr_dicts in paper_corr_index.items():
            title = paper_titles.get(paper_id, "Unknown Paper")
            paper_claims = self._format_paper_claims(corr_dicts)

            # T1: Initial generation — skip paper on LLM failure
            try:
                llm_result = await self._generator.generate(
                    title=title,
                    paper_claims=paper_claims,
                    correlations=corr_dicts,
                    query=query,
                    round_num=0,
                    general_content=formatted_general,
                )
            except Exception as exc:
                errors.append(HypothesisError(
                    message=(
                        f"Initial generation failed for paper '{title}': {exc}"
                    ),
                    exception_type=type(exc).__name__,
                ))
                self._logger.warning(
                    "T1: generation failed for paper '%s': %s", title, exc,
                )
                continue

            # T4: Malformed JSON — skip paper's hypotheses
            try:
                parsed = self._parse_hypotheses(llm_result, valid_ids)
            except Exception as exc:
                errors.append(HypothesisError(
                    message=(
                        f"Hypothesis parsing failed for paper '{title}': {exc}"
                    ),
                    exception_type=type(exc).__name__,
                ))
                self._logger.warning(
                    "T4: parse failed for paper '%s': %s", title, exc,
                )
                continue

            # Apply relevance gate
            parsed = [
                h for h in parsed
                if h.scores.relevance >= config.relevance_gate
            ]

            # Cap per paper
            parsed = parsed[:config.max_hypotheses_per_paper]

            all_hypotheses.extend(parsed)

        if not all_hypotheses:
            return HypothesisResult(
                total_input_papers=len(paper_corr_index),
                total_correlations=len(all_corrs),
                errors=errors,
            )

        # --- Step 3: Initial dedup ---
        all_hypotheses = await self._deduplicate(all_hypotheses)

        # --- Step 4: Reflection loop ---
        rounds = 0
        converged = False

        for round_num in range(1, config.max_reflection_rounds + 1):
            before_reflection = all_hypotheses.copy()

            # Pre-dedup for clean input to LLM
            all_hypotheses = await self._deduplicate(all_hypotheses)

            # Run reflection per paper-cluster
            new_hypotheses = await self._run_reflection_round(
                hypotheses=all_hypotheses,
                paper_corr_index=paper_corr_index,
                paper_titles=paper_titles,
                query=query,
                round_num=round_num,
                errors=errors,
                formatted_general=formatted_general,
            )

            # Merge + post-merge dedup
            all_hypotheses.extend(new_hypotheses)
            all_hypotheses = await self._deduplicate(all_hypotheses)

            # Cap total
            all_hypotheses = all_hypotheses[:config.max_total_hypotheses]

            rounds += 1

            # Convergence check
            if self._detect_convergence(before_reflection, all_hypotheses):
                converged = True
                break

            # Short-circuit: no changes at all
            if not new_hypotheses and not any(
                e.message.startswith(f"Reflection R{round_num}")
                for e in errors
            ):
                # No new hypotheses generated AND no errors this round
                # -> unlikely to get more next round
                converged = True
                break

        # --- Step 5: Grounding check ---
        all_hypotheses = self._check_grounding(all_hypotheses, valid_ids)

        # --- Step 6: Assemble result ---
        return HypothesisResult(
            hypotheses=all_hypotheses,
            total_input_papers=len(paper_corr_index),
            total_correlations=len(all_corrs),
            reflection_rounds_completed=rounds,
            converged=converged,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Step 1: Build correlation index
    # ------------------------------------------------------------------

    def _build_correlation_index(
        self,
        all_corrs: list[Correlation | ModuleCorrelation],
    ) -> tuple[dict[str, list[dict]], dict[str, str], set[str]]:
        """Build per-paper correlation index with synthetic IDs (AD-03).

        Returns:
            Tuple of:
            - ``paper_corr_index``: ``{paper_id: [corr_dict, ...]}``
            - ``paper_titles``: ``{paper_id: title}``
            - ``valid_ids``: set of all valid synthetic correlation IDs.
        """
        paper_corr_index: dict[str, list[dict]] = {}
        paper_titles: dict[str, str] = {}
        valid_ids: set[str] = set()

        for corr in all_corrs:
            claim = corr.claim
            paper_id = claim.paper_id

            # Build synthetic ID (AD-03)
            if isinstance(corr, ModuleCorrelation):
                target_name = corr.module.module_name
            else:
                target_name = corr.component.name

            raw = f"{claim.text}|{target_name}"
            synthetic_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
            valid_ids.add(synthetic_id)

            # Derive target description
            if isinstance(corr, ModuleCorrelation):
                target_description = corr.module.summary
                target_type = "module"
            else:
                target_description = corr.component.description
                target_type = "component"

            corr_dict = {
                "id": synthetic_id,
                "claim_text": claim.text,
                "paper_id": paper_id,
                "paper_title": corr.paper_title,
                "target_name": target_name,
                "target_type": target_type,
                "target_description": target_description,
                "correlation_type": corr.correlation_type,
                "reasoning": corr.reasoning,
                "similarity_score": corr.similarity_score,
            }

            paper_corr_index.setdefault(paper_id, []).append(corr_dict)
            if paper_id not in paper_titles:
                paper_titles[paper_id] = corr.paper_title

        return paper_corr_index, paper_titles, valid_ids

    # ------------------------------------------------------------------
    # Step 2 helpers: parse hypotheses from LLM dicts
    # ------------------------------------------------------------------

    def _parse_hypotheses(
        self,
        llm_dicts: list[dict],
        valid_ids: set[str],
    ) -> list[Hypothesis]:
        """Convert LLM-returned dicts to Hypothesis dataclasses.

        Performs manual validation (AD-07: json.loads() + manual).
        Non-conforming entries are silently skipped — pipeline never
        raises (HG-06).
        """
        hypotheses: list[Hypothesis] = []

        for entry in llm_dicts:
            if not isinstance(entry, dict):
                continue

            title = str(entry.get("title", "")).strip()
            description = str(entry.get("description", "")).strip()
            if not title or not description:
                continue

            # Scores
            scores_raw = entry.get("scores", {})
            if isinstance(scores_raw, dict):
                relevance = self._clamp_score(scores_raw.get("relevance", 1))
                feasibility = self._clamp_score(scores_raw.get("feasibility", 1))
                evidence = self._clamp_score(scores_raw.get("evidence", 1))
            else:
                relevance = feasibility = evidence = 1

            scores = HypothesisScores(
                relevance=relevance,
                feasibility=feasibility,
                evidence=evidence,
            )

            # Composite: equal-weighted arithmetic mean
            composite = (relevance + feasibility + evidence) / 3.0

            # Deterministic ID (AD-02)
            hypo_id = hashlib.sha256(
                f"{title}{description}".encode()
            ).hexdigest()[:12]

            # Correlations — filter to only valid synthetic IDs (HG-05)
            raw_corr_ids = entry.get("correlations", [])
            if isinstance(raw_corr_ids, list):
                correlations = [
                    str(cid) for cid in raw_corr_ids if cid in valid_ids
                ]
            else:
                correlations = []

            # Supporting papers
            raw_papers = entry.get("supporting_papers", [])
            supporting_papers = (
                [str(p) for p in raw_papers]
                if isinstance(raw_papers, list)
                else []
            )

            hypotheses.append(
                Hypothesis(
                    id=hypo_id,
                    title=title,
                    description=description,
                    approach=str(entry.get("approach", "")),
                    supporting_papers=supporting_papers,
                    target_metric=str(entry.get("target_metric", "")),
                    expected_improvement=str(
                        entry.get("expected_improvement", "")
                    ),
                    code_changes=str(entry.get("code_changes", "")),
                    scores=scores,
                    composite=composite,
                    correlations=correlations,
                    success_criteria=str(entry.get("success_criteria", "")),
                )
            )

        return hypotheses

    @staticmethod
    def _clamp_score(value: object) -> int:
        """Clamp a score value to the 1-10 integer range."""
        try:
            score = int(float(str(value)))
        except (ValueError, TypeError):
            return 1
        return max(1, min(10, score))

    # ------------------------------------------------------------------
    # Step 3: Deduplication
    # ------------------------------------------------------------------

    async def _deduplicate(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[Hypothesis]:
        """Deduplicate by embedding cosine similarity (AD-06).

        Embeds ``title + description`` for each hypothesis. Pairs with
        similarity >= ``dedup_threshold`` are merged, keeping the higher
        composite score (tie-breaking per AD-06).

        On embedding failure (T3), skips dedup and returns all hypotheses.
        """
        if len(hypotheses) <= 1:
            return hypotheses

        texts = [f"{h.title} {h.description}" for h in hypotheses]

        # T3: Embedding failure → skip dedup
        try:
            vectors = await self._embedder.embed(texts)
        except Exception as exc:
            self._logger.warning(
                "T3: dedup embedding failed, skipping deduplication: %s", exc,
            )
            return hypotheses

        # Compute pairwise cosine similarity matrix
        vecs = np.array(vectors, dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / np.maximum(norms, 1e-10)
        sim_mat = np.dot(normed, normed.T)

        threshold = self._config.dedup_threshold
        to_remove: set[int] = set()

        for i in range(len(hypotheses)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(hypotheses)):
                if j in to_remove:
                    continue
                if float(sim_mat[i, j]) >= threshold:
                    # Keep the one with higher composite score (AD-06)
                    if hypotheses[i].composite >= hypotheses[j].composite:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # i is removed, stop comparing

        return [
            h for idx, h in enumerate(hypotheses) if idx not in to_remove
        ]

    # ------------------------------------------------------------------
    # Step 4: Reflection round
    # ------------------------------------------------------------------

    async def _run_reflection_round(
        self,
        hypotheses: list[Hypothesis],
        paper_corr_index: dict[str, list[dict]],
        paper_titles: dict[str, str],
        query: str,
        round_num: int,
        errors: list[HypothesisError],
        formatted_general: str | None = None,
    ) -> list[Hypothesis]:
        """Run one critique+expand round per paper-cluster (AD-04).

        Sends ALL existing hypotheses (from all papers) to each paper's
        LLM call so the critique has full cross-paper context.
        """
        # Convert to dicts once for all paper clusters
        hypo_dicts = [self._hypothesis_to_dict(h) for h in hypotheses]

        new_hypotheses: list[Hypothesis] = []

        for paper_id, corr_dicts in paper_corr_index.items():
            title = paper_titles.get(paper_id, "Unknown Paper")
            paper_claims = self._format_paper_claims(corr_dicts)

            # T2: Reflection LLM failure → skip this paper, keep going
            try:
                result = await self._generator.critique_and_expand(
                    title=title,
                    paper_claims=paper_claims,
                    correlations=corr_dicts,
                    existing_hypotheses=hypo_dicts,
                    query=query,
                    round_num=round_num,
                    general_content=formatted_general,
                )
            except Exception as exc:
                errors.append(HypothesisError(
                    message=(
                        f"Reflection R{round_num} failed for paper "
                        f"'{title}': {exc}"
                    ),
                    exception_type=type(exc).__name__,
                ))
                self._logger.warning(
                    "T2: reflection R%d failed for paper '%s': %s",
                    round_num, title, exc,
                )
                continue

            # T4: Parse new hypotheses
            new_from_round = result.get("new_hypotheses", [])
            if not isinstance(new_from_round, list):
                new_from_round = []

            # Rebuild valid_ids from this paper's correlations
            valid_ids = {cd["id"] for cd in corr_dicts}

            try:
                parsed = self._parse_hypotheses(new_from_round, valid_ids)
            except Exception as exc:
                self._logger.warning(
                    "T4: parse failed for reflection R%d paper '%s': %s",
                    round_num, title, exc,
                )
                parsed = []

            # Apply relevance gate
            config = self._config
            parsed = [
                h for h in parsed
                if h.scores.relevance >= config.relevance_gate
            ]

            new_hypotheses.extend(parsed)

        return new_hypotheses

    # ------------------------------------------------------------------
    # Step 5: Convergence detection
    # ------------------------------------------------------------------

    def _detect_convergence(
        self,
        before: list[Hypothesis],
        after: list[Hypothesis],
    ) -> bool:
        """Two-phase convergence check (AD-05).

        1. **Title-set stability**: no brand-new titles in ``after``.
        2. **Score stability**: average composite delta < convergence_score_delta.

        Returns True only when BOTH pass.
        """
        old_titles = {h.title for h in before}
        new_titles = {h.title for h in after}

        # New titles emerged → not converged
        if new_titles - old_titles:
            return False

        # Compute score deltas for matching titles
        old_by_title = {h.title: h.composite for h in before}
        deltas: list[float] = []
        for h in after:
            if h.title in old_by_title:
                deltas.append(abs(h.composite - old_by_title[h.title]))

        if not deltas:
            return True  # No matching titles — safe default

        avg_delta = sum(deltas) / len(deltas)
        return avg_delta < self._config.convergence_score_delta

    # ------------------------------------------------------------------
    # Step 6: Grounding check
    # ------------------------------------------------------------------

    def _check_grounding(
        self,
        hypotheses: list[Hypothesis],
        valid_ids: set[str],
    ) -> list[Hypothesis]:
        """Filter out ungrounded hypotheses (HG-05, D4).

        Each hypothesis must reference at least one valid correlation ID.
        Ungrounded hypotheses are discarded with a warning.
        """
        grounded: list[Hypothesis] = []
        for h in hypotheses:
            if any(cid in valid_ids for cid in h.correlations):
                grounded.append(h)
            else:
                self._logger.warning(
                    "Discarding ungrounded hypothesis '%s' (no valid "
                    "correlation IDs)",
                    h.title,
                )
        return grounded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_paper_claims(corr_dicts: list[dict]) -> str:
        """Format correlation claims as a numbered list.

        Deduplicated by claim_text so repeated correlations across
        components/modules don't appear multiple times.
        """
        seen: set[str] = set()
        lines: list[str] = []
        for cd in corr_dicts:
            text = cd.get("claim_text", "")
            if text and text not in seen:
                seen.add(text)
                lines.append(f"- {text}")
        return "\n".join(lines) if lines else "(no claims)"

    @staticmethod
    def _format_general_content(
        items: list[GeneralContent],
        *,
        max_items: int = 5,
        max_body_chars: int = 500,
    ) -> str | None:
        """Format GeneralContent items for prompt injection.

        Each item gets a titled entry with the body truncated to
        ``max_body_chars``. At most ``max_items`` entries are included
        to avoid blowing up the prompt.

        Returns ``None`` for an empty list — callers can safely pass
        the result to the generator without extra checks.
        """
        if not items:
            return None

        lines: list[str] = []
        for i, gc in enumerate(items[:max_items]):
            body = gc.body.strip()
            url_label = f" ({gc.url})" if gc.url else ""
            truncated = (
                body[:max_body_chars] + "..."
                if len(body) > max_body_chars
                else body
            )
            lines.append(f"[{i + 1}] {gc.title}{url_label}")
            if truncated:
                lines.append(f"    {truncated}")

        return "\n".join(lines) if lines else None

    @staticmethod
    def _hypothesis_to_dict(h: Hypothesis) -> dict:
        """Convert a Hypothesis back to a dict for LLM consumption."""
        return {
            "id": h.id,
            "title": h.title,
            "description": h.description,
            "approach": h.approach,
            "supporting_papers": h.supporting_papers,
            "target_metric": h.target_metric,
            "expected_improvement": h.expected_improvement,
            "code_changes": h.code_changes,
            "scores": {
                "relevance": h.scores.relevance,
                "feasibility": h.scores.feasibility,
                "evidence": h.scores.evidence,
            },
            "composite": h.composite,
            "correlations": h.correlations,
            "success_criteria": h.success_criteria,
        }
