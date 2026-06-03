"""Correlation pipeline — embed → cosine → filter → classify → assemble.

Hybrid approach (AD-01): embedding similarity pre-filters cheaply, then
a batched LLM call per paper classifies correlation types with reasoning
(D2, D9). The pipeline never raises — every stage degrades gracefully
following the 4-tier error resilience cascade (CM-05, AD-05).
"""

from __future__ import annotations

import logging

import numpy as np

from research_to_dev.codebase.types import CodebaseContext, Component, ModuleSummary
from research_to_dev.paper_profile.types import Claim, PaperProfile
from research_to_dev.ranking.protocols import Embedder
from research_to_dev.shared.config import CorrelationMapConfig

from .protocols import CorrelationClassifier
from .types import Correlation, CorrelationError, CorrelationMap, ModuleCorrelation


class CorrelationPipeline:
    """Orchestrates the full embed→cosine→filter→classify→assemble flow.

    Reuses the shared ``Embedder`` Protocol from ``ranking.protocols``
    (no duplication — D11). The ``CorrelationClassifier`` is a new
    Protocol because the classification task differs from ranking's
    ``LLMJudge`` (D12).

    All dependencies are constructor-injected so the pipeline is fully
    testable with mock embedders and classifiers.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        classifier: CorrelationClassifier,
        config: CorrelationMapConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the pipeline with injected dependencies.

        Args:
            embedder: Shared ``Embedder`` Protocol implementation (e.g.
                ``OpenAIEmbedder`` from the ranking module).
            classifier: ``CorrelationClassifier`` Protocol implementation
                (e.g. ``OpenAICorrelationClassifier``).
            config: Configuration dataclass. Defaults to
                ``CorrelationMapConfig()``.
            logger: Logger instance. Defaults to
                ``logging.getLogger(__name__)``.
        """
        self._embedder = embedder
        self._classifier = classifier
        self._config = config or CorrelationMapConfig()
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def correlate(
        self,
        profiles: list[PaperProfile],
        codebase: CodebaseContext,
    ) -> CorrelationMap:
        """Run the full correlation pipeline.

        Args:
            profiles: List of paper profiles with claims to match.
            codebase: Codebase context with components and module summaries.

        Returns:
            A ``CorrelationMap`` with validated matches. Never raises —
            returns a degraded (possibly empty) map on failure (CM-05).
        """
        config = self._config

        # --- Guard: empty inputs ---
        components = codebase.components
        modules = codebase.modules
        all_claims: list[tuple[Claim, str]] = []
        for profile in profiles:
            for claim in profile.claims:
                paper_title = profile.ranked_paper.paper.title
                all_claims.append((claim, paper_title))

        if not all_claims or (not components and not modules):
            return CorrelationMap(
                papers_analyzed=len(profiles),
                total_claims=len(all_claims),
                total_components=len(components),
                total_modules=len(modules),
                similarity_threshold=config.similarity_threshold,
                embedding_model=config.embedding_model,
            )

        # --- Stage 1: Embedding (CM-01) ---
        try:
            claim_vecs, comp_vecs, mod_vecs = await self._embed_all(
                all_claims, components, modules, config
            )
        except Exception as exc:
            self._logger.warning("Embedding stage failed: %s", exc)
            return self._empty_map(len(profiles), len(all_claims), components, modules, config)

        # --- Stage 2: Cosine similarity + threshold filtering (CM-02) ---
        try:
            comp_candidates, mod_candidates = self._compute_candidates(
                all_claims, claim_vecs,
                components, comp_vecs,
                modules, mod_vecs,
                config.similarity_threshold,
            )
        except Exception as exc:
            self._logger.warning("Cosine similarity stage failed: %s", exc)
            return self._empty_map(len(profiles), len(all_claims), components, modules, config)

        # --- Stage 3: LLM classification per paper (CM-03) ---
        all_classified = await self._classify_by_paper(
            all_claims, comp_candidates, mod_candidates, components, modules,
        )

        # --- Stage 4: Assemble CorrelationMap (CM-04) ---
        return self._assemble(
            all_classified, all_claims, profiles, components, modules, config,
        )

    # ------------------------------------------------------------------
    # Stage 1: Embed all texts
    # ------------------------------------------------------------------

    async def _embed_all(
        self,
        all_claims: list[tuple[Claim, str]],
        components: list[Component],
        modules: list[ModuleSummary],
        config: CorrelationMapConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embed claim texts, component descriptions, and module summaries.

        Returns three numpy arrays of shape (n, d) where d is the
        embedding dimension.
        """
        claim_texts = [claim.text for claim, _ in all_claims]
        comp_texts = [c.description for c in components]
        mod_texts = [m.summary for m in modules]  # AD-04: summary only

        all_texts = claim_texts + comp_texts + mod_texts

        # Filter out empty texts to avoid embedding API errors
        non_empty_indices = [i for i, t in enumerate(all_texts) if t.strip()]
        non_empty_texts = [all_texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            # All texts are empty — return zero vectors
            dim = 1536  # text-embedding-3-small default
            n_claims = len(claim_texts)
            n_comps = len(comp_texts)
            n_mods = len(mod_texts)
            return (
                np.zeros((n_claims, dim)),
                np.zeros((n_comps, dim)),
                np.zeros((n_mods, dim)),
            )

        vectors = await self._embedder.embed(non_empty_texts)

        # Reconstruct full arrays with zeros for empty texts
        dim = len(vectors[0]) if vectors else 1536
        n_claims = len(claim_texts)
        n_comps = len(comp_texts)
        n_mods = len(mod_texts)

        full_vecs = np.zeros((len(all_texts), dim))
        for idx, vec in zip(non_empty_indices, vectors):
            full_vecs[idx] = np.array(vec)

        claim_vecs = full_vecs[:n_claims]
        comp_vecs = full_vecs[n_claims:n_claims + n_comps]
        mod_vecs = full_vecs[n_claims + n_comps:]

        return claim_vecs, comp_vecs, mod_vecs

    # ------------------------------------------------------------------
    # Stage 2: Cosine similarity + threshold filter
    # ------------------------------------------------------------------

    def _compute_candidates(
        self,
        all_claims: list[tuple[Claim, str]],
        claim_vecs: np.ndarray,
        components: list[Component],
        comp_vecs: np.ndarray,
        modules: list[ModuleSummary],
        mod_vecs: np.ndarray,
        threshold: float,
    ) -> tuple[list[dict], list[dict]]:
        """Compute cosine similarity matrices and filter by threshold.

        Returns two lists of candidate dicts for the LLM classifier.
        """
        comp_candidates: list[dict] = []
        mod_candidates: list[dict] = []

        # Claim × Component matrix
        if len(components) > 0 and claim_vecs.size > 0 and comp_vecs.size > 0:
            comp_sim = _cosine_matrix(claim_vecs, comp_vecs)
            for i, (claim, paper_title) in enumerate(all_claims):
                for j, component in enumerate(components):
                    score = float(comp_sim[i, j])
                    if score >= threshold:
                        comp_candidates.append({
                            "claim_index": i,
                            "claim_text": claim.text,
                            "paper_title": paper_title,
                            "paper_id": claim.paper_id,
                            "similarity_score": score,
                            "target_type": "component",
                            "target_text": component.description,
                            "target_name": component.name,
                            "target_index": j,
                            "module_name": component.module_name,
                        })

        # Claim × Module matrix
        if len(modules) > 0 and claim_vecs.size > 0 and mod_vecs.size > 0:
            mod_sim = _cosine_matrix(claim_vecs, mod_vecs)
            for i, (claim, paper_title) in enumerate(all_claims):
                for j, mod_summary in enumerate(modules):
                    score = float(mod_sim[i, j])
                    if score >= threshold:
                        mod_candidates.append({
                            "claim_index": i,
                            "claim_text": claim.text,
                            "paper_title": paper_title,
                            "paper_id": claim.paper_id,
                            "similarity_score": score,
                            "target_type": "module",
                            "target_text": mod_summary.summary,
                            "target_name": mod_summary.module_name,
                            "target_index": j,
                            "module_name": mod_summary.module_name,
                        })

        return comp_candidates, mod_candidates

    # ------------------------------------------------------------------
    # Stage 3: Batched LLM classification per paper
    # ------------------------------------------------------------------

    async def _classify_by_paper(
        self,
        all_claims: list[tuple[Claim, str]],
        comp_candidates: list[dict],
        mod_candidates: list[dict],
        components: list[Component],
        modules: list[ModuleSummary],
    ) -> list[dict]:
        """Group candidates by paper and classify each batch (D9, CM-03).

        Silently discards LLM-rejected matches (D10). On per-paper LLM
        failure, that paper's candidates are skipped (CM-05, AD-05).
        """
        # Merge both candidate lists
        all_candidates = comp_candidates + mod_candidates

        # Group by paper_id
        by_paper: dict[str, list[dict]] = {}
        for c in all_candidates:
            pid = c["paper_id"]
            by_paper.setdefault(pid, []).append(c)

        all_validated: list[dict] = []

        for paper_id, candidates in by_paper.items():
            if not candidates:
                continue

            # Derive title from first candidate
            title = candidates[0]["paper_title"]

            # Collect all claims for this paper (deduplicated by claim_index)
            seen_indices: set[int] = set()
            paper_claim_texts: list[str] = []
            for c in candidates:
                idx = c["claim_index"]
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    paper_claim_texts.append(f"- {all_claims[idx][0].text}")

            paper_claims_str = "\n".join(paper_claim_texts)

            # Build classifier-compatible candidate dicts
            classifier_candidates = [
                {
                    "claim_text": c["claim_text"],
                    "similarity_score": c["similarity_score"],
                    "target_text": c["target_text"],
                    "target_type": c["target_type"],
                    "target_name": c["target_name"],
                    "module_name": c["module_name"],
                }
                for c in candidates
            ]

            try:
                validated = await self._classifier.classify(
                    title=title,
                    paper_claims=paper_claims_str,
                    candidates=classifier_candidates,
                )
            except Exception as exc:
                self._logger.warning(
                    "LLM classification failed for paper '%s': %s", title, exc
                )
                continue

            # Enrich validated matches with the original candidate data
            for match in validated:
                match_text = match.get("claim_text", "")
                match_target = match.get("target_name", "")
                # Find the original candidate to recover metadata
                for cand in candidates:
                    if (
                        cand["claim_text"] == match_text
                        and cand["target_name"] == match_target
                    ):
                        match["_claim_index"] = cand["claim_index"]
                        match["_paper_title"] = cand["paper_title"]
                        match["_similarity_score"] = cand["similarity_score"]
                        match["_target_type"] = cand["target_type"]
                        match["_target_index"] = cand["target_index"]
                        match["_module_name"] = cand["module_name"]
                        match["_paper_id"] = cand["paper_id"]
                        break
                all_validated.append(match)

        return all_validated

    # ------------------------------------------------------------------
    # Stage 4: Assemble CorrelationMap
    # ------------------------------------------------------------------

    def _assemble(
        self,
        validated: list[dict],
        all_claims: list[tuple[Claim, str]],
        profiles: list[PaperProfile],
        components: list[Component],
        modules: list[ModuleSummary],
        config: CorrelationMapConfig,
    ) -> CorrelationMap:
        """Build the CorrelationMap from LLM-validated matches (CM-04)."""
        correlations: list[Correlation] = []
        module_correlations: list[ModuleCorrelation] = []

        for match in validated:
            target_type = match.get("_target_type", "")
            claim_idx = match.get("_claim_index")
            target_idx = match.get("_target_index")
            if claim_idx is None or target_idx is None:
                continue

            claim, _ = all_claims[claim_idx]

            correlation_type = match.get("correlation_type", "unclassified")
            reasoning = match.get("reasoning", "")
            paper_title = match.get("_paper_title", "")
            similarity_score = float(match.get("_similarity_score", 0.0))

            if target_type == "component" and target_idx < len(components):
                comp = components[target_idx]
                correlations.append(
                    Correlation(
                        claim=claim,
                        paper_title=paper_title,
                        component=comp,
                        module_name=comp.module_name,
                        similarity_score=similarity_score,
                        correlation_type=correlation_type,
                        reasoning=reasoning,
                    )
                )
            elif target_type == "module" and target_idx < len(modules):
                mod = modules[target_idx]
                module_correlations.append(
                    ModuleCorrelation(
                        claim=claim,
                        paper_title=paper_title,
                        module=mod,
                        similarity_score=similarity_score,
                        correlation_type=correlation_type,
                        reasoning=reasoning,
                    )
                )

        return CorrelationMap(
            correlations=correlations,
            module_correlations=module_correlations,
            papers_analyzed=len(profiles),
            total_claims=len(all_claims),
            total_components=len(components),
            total_modules=len(modules),
            matches_found=len(correlations) + len(module_correlations),
            similarity_threshold=config.similarity_threshold,
            embedding_model=config.embedding_model,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_map(
        self,
        papers_analyzed: int,
        total_claims: int,
        components: list[Component],
        modules: list[ModuleSummary],
        config: CorrelationMapConfig,
    ) -> CorrelationMap:
        """Return a degraded empty CorrelationMap with accurate counts (CM-05)."""
        return CorrelationMap(
            papers_analyzed=papers_analyzed,
            total_claims=total_claims,
            total_components=len(components),
            total_modules=len(modules),
            similarity_threshold=config.similarity_threshold,
            embedding_model=config.embedding_model,
        )


# ======================================================================
# Module-level helpers
# ======================================================================


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity matrix between two sets of vectors.

    Uses numpy-only operations (AD-03): a·b / (‖a‖·‖b‖).

    Args:
        a: Matrix of shape ``(n, d)`` — e.g. claim vectors.
        b: Matrix of shape ``(m, d)`` — e.g. component/module vectors.

    Returns:
        Matrix of shape ``(n, m)`` where ``result[i, j]`` is the cosine
        similarity between ``a[i]`` and ``b[j]``.
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = a / np.maximum(a_norm, 1e-10)
    b_unit = b / np.maximum(b_norm, 1e-10)
    return np.dot(a_unit, b_unit.T)
