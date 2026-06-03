#!/usr/bin/env python3
"""End-to-end test of the CorrelationMap pipeline with real OpenAI API calls.

Validates the full embed→cosine→filter→classify→assemble flow
using realistic sample data (3 ML papers + 8 code components + 3 modules).

Requires: OPENAI_API_KEY set in a ``.env`` file at the project root.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure the package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Load OPENAI_API_KEY from .env (manual parsing — no extra deps needed)
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print("ERROR: .env file not found at", env_path)
        sys.exit(1)

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "OPENAI_API_KEY":
            val = value.strip().strip('"').strip("'")
            if val:
                return val

    print("ERROR: OPENAI_API_KEY not found in .env")
    sys.exit(1)


OPENAI_API_KEY = _load_api_key()
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Now import project modules (API key must be in env before AsyncOpenAI init)
# ---------------------------------------------------------------------------
from openai import AsyncOpenAI

from research_to_dev.codebase.types import CodebaseContext, Component, ModuleSummary
from research_to_dev.correlation.adapters import OpenAICorrelationClassifier
from research_to_dev.correlation.pipeline import CorrelationPipeline
from research_to_dev.correlation.types import Correlation, CorrelationMap, ModuleCorrelation
from research_to_dev.extraction.types import AcademicContent
from research_to_dev.paper_profile.types import Claim, PaperProfile
from research_to_dev.ranking.adapters import OpenAIEmbedder
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import CorrelationMapConfig


# ======================================================================
# Sample Data
# ======================================================================

def build_paper_profiles() -> list[PaperProfile]:
    """Construct 3 realistic PaperProfile objects with ML claims."""

    # --- Paper 1: Attention Is All You Need ---
    paper1 = AcademicContent(
        source="arxiv",
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        abstract=(
            "The dominant sequence transduction models are based on complex "
            "recurrent or convolutional neural networks. We propose a new "
            "simple network architecture, the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions "
            "entirely. Experiments show the Transformer generalizes well to "
            "other tasks."
        ),
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        published_date="2017-06-12",
    )
    ranked1 = RankedPaper(
        paper=paper1,
        relevance_score=0.95,
        semantic_score=0.88,
        llm_reasoning="Directly relevant: proposes the Transformer architecture used in modern ML.",
        passed_metadata_filter=True,
    )
    profile1 = PaperProfile(
        ranked_paper=ranked1,
        claims=[
            Claim(
                text="Self-attention mechanism computes representation by "
                "attending to all positions in the input simultaneously.",
                paper_id="attention-is-all-you-need-2017",
                section_name="method",
            ),
            Claim(
                text="Multi-head attention allows the model to jointly attend "
                "to information from different representation subspaces.",
                paper_id="attention-is-all-you-need-2017",
                section_name="method",
            ),
            Claim(
                text="Positional encodings are added to input embeddings to "
                "inject information about token position in the sequence.",
                paper_id="attention-is-all-you-need-2017",
                section_name="method",
            ),
            Claim(
                text="Layer normalization is applied after each sub-layer "
                "followed by residual connections.",
                paper_id="attention-is-all-you-need-2017",
                section_name="method",
            ),
        ],
    )

    # --- Paper 2: Batch Normalization ---
    paper2 = AcademicContent(
        source="arxiv",
        title="Batch Normalization: Accelerating Deep Network Training by "
        "Reducing Internal Covariate Shift",
        url="https://arxiv.org/abs/1502.03167",
        abstract=(
            "Training Deep Neural Networks is complicated by the fact that "
            "the distribution of each layer's inputs changes during training, "
            "as the parameters of the previous layers change. We refer to this "
            "phenomenon as internal covariate shift, and propose Batch "
            "Normalization to address it."
        ),
        authors=["Sergey Ioffe", "Christian Szegedy"],
        published_date="2015-02-11",
    )
    ranked2 = RankedPaper(
        paper=paper2,
        relevance_score=0.90,
        semantic_score=0.85,
        llm_reasoning="Relevant: batch normalization is a core training technique.",
        passed_metadata_filter=True,
    )
    profile2 = PaperProfile(
        ranked_paper=ranked2,
        claims=[
            Claim(
                text="Batch normalization normalizes activations using the "
                "mean and variance of the current mini-batch.",
                paper_id="batch-normalization-2015",
                section_name="method",
            ),
            Claim(
                text="Internal covariate shift is the change in distribution "
                "of network activations due to the change in network "
                "parameters during training.",
                paper_id="batch-normalization-2015",
                section_name="introduction",
            ),
            Claim(
                text="Batch normalization enables higher learning rates and "
                "reduces the need for careful initialization, acting as a "
                "regularizer.",
                paper_id="batch-normalization-2015",
                section_name="results",
            ),
        ],
    )

    # --- Paper 3: Dropout ---
    paper3 = AcademicContent(
        source="arxiv",
        title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        url="https://jmlr.org/papers/v15/srivastava14a.html",
        abstract=(
            "Deep neural nets with a large number of parameters are very "
            "powerful machine learning systems. However, overfitting is a "
            "serious problem in such networks. We present dropout, a technique "
            "for addressing this problem."
        ),
        authors=["Nitish Srivastava", "Geoffrey Hinton", "Alex Krizhevsky"],
        published_date="2014-06-01",
    )
    ranked3 = RankedPaper(
        paper=paper3,
        relevance_score=0.85,
        semantic_score=0.82,
        llm_reasoning="Relevant: dropout is a standard regularization technique.",
        passed_metadata_filter=True,
    )
    profile3 = PaperProfile(
        ranked_paper=ranked3,
        claims=[
            Claim(
                text="Dropout randomly drops units and their connections "
                "during training to prevent overfitting.",
                paper_id="dropout-2014",
                section_name="method",
            ),
            Claim(
                text="At test time, all units are used with their outgoing "
                "weights multiplied by the dropout probability, approximating "
                "a model averaging effect.",
                paper_id="dropout-2014",
                section_name="method",
            ),
            Claim(
                text="Overfitting occurs when a model learns the training data "
                "too well, including noise, resulting in poor generalization.",
                paper_id="dropout-2014",
                section_name="introduction",
            ),
        ],
    )

    return [profile1, profile2, profile3]


def build_codebase_context() -> CodebaseContext:
    """Construct a realistic ML project codebase with components and modules."""

    components = [
        Component(
            name="AttentionLayer",
            kind="class",
            file_path="models/attention.py",
            line_start=10,
            line_end=85,
            signature="class AttentionLayer(nn.Module)",
            description=(
                "Implements multi-head scaled dot-product attention. Projects "
                "queries, keys, and values through linear layers, computes "
                "attention scores via softmax, and concatenates multi-head "
                "outputs. Includes optional causal masking for autoregressive "
                "decoding."
            ),
            module_name="models.attention",
        ),
        Component(
            name="batch_normalize",
            kind="function",
            file_path="training/pipeline.py",
            line_start=45,
            line_end=68,
            signature="def batch_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor",
            description=(
                "Applies batch normalization to input tensor x by computing "
                "mean and variance across the batch dimension. Uses running "
                "statistics during inference mode."
            ),
            module_name="training.pipeline",
        ),
        Component(
            name="dropout_forward",
            kind="function",
            file_path="training/pipeline.py",
            line_start=72,
            line_end=95,
            signature="def dropout_forward(x: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor",
            description=(
                "Applies dropout regularization during training by randomly "
                "zeroing elements with probability p. During evaluation, "
                "scales outputs by (1-p) to maintain expected values."
            ),
            module_name="training.pipeline",
        ),
        Component(
            name="train_model",
            kind="function",
            file_path="training/pipeline.py",
            line_start=100,
            line_end=165,
            signature="def train_model(model, dataloader, optimizer, criterion, epochs: int, device: str = 'cuda')",
            description=(
                "Main training loop that iterates over epochs, runs forward "
                "passes, computes loss via the given criterion, backpropagates "
                "gradients, and applies optimizer steps. Includes checkpoint "
                "saving and early stopping."
            ),
            module_name="training.pipeline",
        ),
        Component(
            name="evaluate",
            kind="function",
            file_path="training/pipeline.py",
            line_start=170,
            line_end=200,
            signature="def evaluate(model, dataloader, metrics: list[str]) -> dict[str, float]",
            description=(
                "Evaluates model performance on a validation dataset. "
                "Computes metrics including accuracy, precision, recall, and "
                "F1 score. Runs in no_grad mode for efficiency."
            ),
            module_name="training.pipeline",
        ),
        Component(
            name="preprocess_data",
            kind="function",
            file_path="data/preprocessing.py",
            line_start=15,
            line_end=60,
            signature="def preprocess_data(raw_data: list[dict], tokenizer, max_length: int = 512) -> Dataset",
            description=(
                "Preprocesses raw text data by tokenizing, truncating to "
                "max_length, adding special tokens, and creating attention "
                "masks. Returns a HuggingFace Dataset ready for training."
            ),
            module_name="data.preprocessing",
        ),
        Component(
            name="save_checkpoint",
            kind="function",
            file_path="training/pipeline.py",
            line_start=205,
            line_end=230,
            signature="def save_checkpoint(model, optimizer, epoch: int, path: str, best_loss: float)",
            description=(
                "Saves model state dict, optimizer state, current epoch, and "
                "best loss to a checkpoint file. Supports resuming training "
                "from any checkpoint."
            ),
            module_name="training.pipeline",
        ),
        Component(
            name="DataLoader",
            kind="class",
            file_path="data/preprocessing.py",
            line_start=65,
            line_end=110,
            signature="class DataLoader",
            description=(
                "Custom DataLoader that wraps PyTorch's DataLoader with "
                "batching, shuffling, and prefetching. Handles variable-length "
                "sequences with padding and collation."
            ),
            module_name="data.preprocessing",
        ),
    ]

    modules = [
        ModuleSummary(
            module_name="training.pipeline",
            file_path="training/pipeline.py",
            summary=(
                "Core training pipeline module containing the main training "
                "loop, normalization utilities, regularization (dropout), "
                "evaluation, and checkpoint management."
            ),
            key_responsibilities=[
                "Orchestrate model training across epochs",
                "Apply batch normalization during forward passes",
                "Apply dropout regularization during training",
                "Evaluate model performance on validation data",
                "Save and resume from training checkpoints",
            ],
        ),
        ModuleSummary(
            module_name="models.attention",
            file_path="models/attention.py",
            summary=(
                "Attention module implementing multi-head scaled dot-product "
                "attention with projection layers, optional masking, and "
                "residual connections."
            ),
            key_responsibilities=[
                "Compute multi-head attention scores",
                "Apply causal masking for autoregressive tasks",
                "Project queries, keys, and values to multiple heads",
                "Concatenate and project multi-head outputs",
            ],
        ),
        ModuleSummary(
            module_name="data.preprocessing",
            file_path="data/preprocessing.py",
            summary=(
                "Data preprocessing module handling tokenization, batching, "
                "and dataset creation for ML training pipelines."
            ),
            key_responsibilities=[
                "Tokenize and truncate raw text data",
                "Create PyTorch datasets with attention masks",
                "Provide batching with variable-length sequence support",
            ],
        ),
    ]

    return CodebaseContext(
        project_name="ml-research-project",
        project_path="/home/user/ml-research-project",
        language="python",
        components=components,
        modules=modules,
        total_files_scanned=12,
        total_components_found=8,
    )


# ======================================================================
# Report formatting
# ======================================================================

def print_header(text: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {text}")
    print(f"{'=' * 72}")


def print_subheader(text: str) -> None:
    print(f"\n{'-' * 72}")
    print(f"  {text}")
    print(f"{'-' * 72}")


def truncate(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def print_report(corrmap: CorrelationMap, profiles: list[PaperProfile]) -> None:
    """Print a human-readable report of the correlation results."""

    # --- Summary stats ---
    print_header("CORRELATIONMAP E2E REPORT")
    print(f"\n  Config: similarity_threshold={corrmap.similarity_threshold}, "
          f"embedding_model={corrmap.embedding_model}")
    print(f"\n  Papers analyzed:  {corrmap.papers_analyzed}")
    print(f"  Total claims:     {corrmap.total_claims}")
    print(f"  Total components: {corrmap.total_components}")
    print(f"  Total modules:    {corrmap.total_modules}")
    print(f"  Matches found:    {corrmap.matches_found}")
    print(f"    - Component correlations: {len(corrmap.correlations)}")
    print(f"    - Module correlations:    {len(corrmap.module_correlations)}")

    # --- Component correlations ---
    if corrmap.correlations:
        print_subheader("COMPONENT CORRELATIONS")
        corr: Correlation
        for i, corr in enumerate(corrmap.correlations, 1):
            print(f"\n  [{i}] {corr.correlation_type}")
            print(f"      Paper:   {corr.paper_title}")
            print(f"      Section: {corr.claim.section_name}")
            print(f"      Claim:   \"{truncate(corr.claim.text)}\"")
            print(f"      Target:  {corr.component.name!r} "
                  f"(in {corr.module_name})")
            print(f"      Score:   {corr.similarity_score:.4f}")
            print(f"      Reason:  {truncate(corr.reasoning, 120)}")
    else:
        print_subheader("COMPONENT CORRELATIONS — None found")

    # --- Module correlations ---
    if corrmap.module_correlations:
        print_subheader("MODULE CORRELATIONS")
        mcorr: ModuleCorrelation
        for i, mcorr in enumerate(corrmap.module_correlations, 1):
            print(f"\n  [{i}] {mcorr.correlation_type}")
            print(f"      Paper:   {mcorr.paper_title}")
            print(f"      Section: {mcorr.claim.section_name}")
            print(f"      Claim:   \"{truncate(mcorr.claim.text)}\"")
            print(f"      Target:  {mcorr.module.module_name!r}")
            print(f"      Score:   {mcorr.similarity_score:.4f}")
            print(f"      Reason:  {truncate(mcorr.reasoning, 120)}")
    else:
        print_subheader("MODULE CORRELATIONS — None found")

    # --- Stats summary ---
    print_header("STATS SUMMARY")

    if corrmap.correlations:
        types = {}
        for c in corrmap.correlations:
            types[c.correlation_type] = types.get(c.correlation_type, 0) + 1
        print("\n  Correlation types (component):")
        for t, count in sorted(types.items()):
            print(f"    - {t}: {count}")

    if corrmap.module_correlations:
        mtypes: dict[str, int] = {}
        for m in corrmap.module_correlations:
            mtypes[m.correlation_type] = mtypes.get(m.correlation_type, 0) + 1
        print("\n  Correlation types (module):")
        for t, count in sorted(mtypes.items()):
            print(f"    - {t}: {count}")

    by_paper: dict[str, int] = {}
    for c in corrmap.correlations:
        by_paper[c.paper_title] = by_paper.get(c.paper_title, 0) + 1
    for m in corrmap.module_correlations:
        by_paper[m.paper_title] = by_paper.get(m.paper_title, 0) + 1

    if by_paper:
        print("\n  Matches per paper:")
        for title, count in sorted(by_paper.items(), key=lambda x: -x[1]):
            print(f"    - {truncate(title, 60)}: {count}")

    paper_names = [p.ranked_paper.paper.title for p in profiles]
    for name in paper_names:
        if name not in by_paper:
            print(f"    - {truncate(name, 60)}: 0")

    # Trust: did we get any matches at all?
    print_header("RESULT")
    if corrmap.matches_found > 0:
        print(f"\n  ✅ SUCCESS: CorrelationMap found {corrmap.matches_found} "
              f"validated matches with real OpenAI embeddings + LLM.\n")
    else:
        print(f"\n  ⚠️  NO MATCHES: Pipeline ran but the LLM rejected all "
              f"candidates. The similarity threshold ({corrmap.similarity_threshold}) "
              f"may be too high for the sample data, or the LLM was conservative.\n")
    print(f"{'=' * 72}\n")


# ======================================================================
# Main
# ======================================================================

async def main() -> None:
    # Enable logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print_header("CORRELATIONMAP E2E — Real OpenAI API Calls")
    print("\n  ⏳ Loading sample data...")

    profiles = build_paper_profiles()
    codebase = build_codebase_context()

    print(f"  ✅ {len(profiles)} paper profiles loaded")
    print(f"  ✅ {len(codebase.components)} components loaded")
    print(f"  ✅ {len(codebase.modules)} modules loaded")

    # --- Create real OpenAI clients ---
    print("\n  ⏳ Initialising OpenAI clients...")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    embedder = OpenAIEmbedder(client=client, model="text-embedding-3-small")
    classifier = OpenAICorrelationClassifier(client=client, model="gpt-4o-mini")
    config = CorrelationMapConfig()

    # --- Pre-run cosine analysis (debug: check if candidates pass threshold) ---
    print("\n  ⏳ Computing cosine similarity diagnostics...")
    all_claims: list[str] = []
    for profile in profiles:
        for claim in profile.claims:
            all_claims.append(claim.text)

    comp_texts = [c.description for c in codebase.components]
    mod_texts = [m.summary for m in codebase.modules]

    # Batch embed everything in one call
    all_texts = all_claims + comp_texts + mod_texts
    vectors = await embedder.embed(all_texts)

    n_c = len(all_claims)
    n_comp = len(comp_texts)
    n_mod = len(mod_texts)

    import numpy as np
    claim_vecs = np.array(vectors[:n_c])
    comp_vecs = np.array(vectors[n_c:n_c + n_comp])
    mod_vecs = np.array(vectors[n_c + n_comp:])

    threshold = config.similarity_threshold

    # Cosine for claim×component
    print(f"\n  📊 Top claim↔component cosine scores (threshold={threshold}):")
    found_any = False
    for i, ct in enumerate(all_claims):
        for j, cname in enumerate([c.name for c in codebase.components]):
            score = float(np.dot(claim_vecs[i], comp_vecs[j]) /
                          (np.linalg.norm(claim_vecs[i]) * np.linalg.norm(comp_vecs[j]) + 1e-10))
            if score >= threshold:  # show borderline ones too
                marker = "✅" if score >= threshold else "⬜"
                found_any = True
                print(f"     {marker} {score:.4f}  "
                      f"\"{truncate(ct, 50)}\"  ↔  {cname!r}")

    if not found_any:
        print(f"     (none ≥ {threshold})")

    # Cosine for claim×module
    print(f"\n  📊 Top claim↔module cosine scores (threshold={threshold}):")
    found_mod = False
    for i, ct in enumerate(all_claims):
        for j, mname in enumerate([m.module_name for m in codebase.modules]):
            score = float(np.dot(claim_vecs[i], mod_vecs[j]) /
                          (np.linalg.norm(claim_vecs[i]) * np.linalg.norm(mod_vecs[j]) + 1e-10))
            if score >= threshold:
                marker = "✅" if score >= threshold else "⬜"
                found_mod = True
                print(f"     {marker} {score:.4f}  "
                      f"\"{truncate(ct, 50)}\"  ↔  {mname!r}")

    if not found_mod:
        print(f"     (none ≥ {threshold})")

    # --- Run pipeline ---
    pipeline = CorrelationPipeline(
        embedder=embedder,
        classifier=classifier,
        config=config,
    )

    print(f"\n  ℹ️  Pipeline similarity_threshold = {config.similarity_threshold}")
    print("\n  ⏳ Running CorrelationMap pipeline...")
    print("      → Embedding claims + components + modules (1 API call)")
    print("      → Cosine similarity filtering")
    print("      → Batched LLM classification per paper (up to 3 API calls)")
    print("      → Assembling CorrelationMap...")

    # --- Run ---
    corrmap: CorrelationMap = await pipeline.correlate(
        profiles=profiles,
        codebase=codebase,
    )

    # --- Report ---
    print_report(corrmap, profiles)


if __name__ == "__main__":
    asyncio.run(main())
