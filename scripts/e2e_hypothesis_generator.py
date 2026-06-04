#!/usr/bin/env python3
"""End-to-end test: CorrelationMap -> HypothesisGenerator with real OpenAI API calls.

Chains the existing CorrelationMap sample data through:
  1. CorrelationMap pipeline (embed -> cosine -> filter -> classify -> assemble)
  2. HypothesisGenerator pipeline (generate -> reflect -> score -> ground -> rank)

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
# Load OPENAI_API_KEY from .env (manual parsing -- no extra deps needed)
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
from research_to_dev.extraction.types import AcademicContent, GeneralContent
from research_to_dev.hypothesis.adapters import OpenAIHypothesisGenerator
from research_to_dev.hypothesis.pipeline import HypothesisPipeline
from research_to_dev.hypothesis.types import Hypothesis, HypothesisResult
from research_to_dev.paper_profile.types import Claim, PaperProfile
from research_to_dev.ranking.adapters import OpenAIEmbedder
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import CorrelationMapConfig, HypothesisConfig


# ======================================================================
# Sample Data (copied from e2e_correlation_map.py for standalone use)
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


def build_general_content() -> list[GeneralContent]:
    """Construct 3 realistic GeneralContent items for supplementary context."""
    return [
        GeneralContent(
            source="tavily",
            title="Training Stability Tips: Batch Size, LR, and Normalization",
            url="https://blog.mlops.com/training-stability-tips-2024",
            body=(
                "Training deep neural networks is often plagued by instability "
                "issues like exploding gradients, vanishing gradients, and "
                "loss spikes. The three most impactful knobs are batch size, "
                "learning rate scheduling, and normalization layers. "
                "Smaller batch sizes (32-128) with gradient clipping at 1.0 "
                "tend to produce more stable training curves. Cosine annealing "
                "with warm restarts (SGDR) often outperforms step decay for "
                "convergence speed. Layer Normalization is preferred over "
                "Batch Normalization when batch sizes are small or variable. "
                "For transformer models specifically, Pre-LN (applying layer "
                "norm before attention/FFN sublayers) dramatically improves "
                "training stability compared to Post-LN. Mixed precision "
                "training (FP16) with loss scaling can also introduce "
                "instability if the loss scale factor is too aggressive. "
                "Monitoring gradient norms per layer is the best way to "
                "detect instability early before it causes NaN losses."
            ),
            metadata={"source_type": "blog", "relevance": 0.92},
        ),
        GeneralContent(
            source="tavily",
            title="PyTorch Performance Tuning Guide — Official Docs",
            url="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
            body=(
                "This guide covers techniques to optimize PyTorch model "
                "training performance. Key recommendations: (1) Use "
                "torch.compile() with mode='reduce-overhead' for 30-50% "
                "speedup on modern GPUs; (2) Enable cudnn.benchmark = True "
                "for fixed input sizes to auto-tune convolution algorithms; "
                "(3) Use pin_memory=True in DataLoader with non_blocking "
                "transfers to overlap data loading with GPU compute; "
                "(4) Fuse operations where possible — LayerNorm + Dropout "
                "can be fused via torch.nn.functional; (5) Gradient "
                "accumulation for larger effective batch sizes when GPU "
                "memory is limited; (6) torch.cuda.amp autocast for mixed "
                "precision training with minimal code changes; "
                "(7) Profile with torch.profiler to identify bottlenecks "
                "before optimizing blindly."
            ),
            metadata={"source_type": "official_docs", "relevance": 0.88},
        ),
        GeneralContent(
            source="tavily",
            title="Gradient Clipping: Why, When, and How Much",
            url="https://towardsdatascience.com/gradient-clipping-explained-2024",
            body=(
                "Gradient clipping is a simple yet effective technique to "
                "stabilize training. It prevents exploding gradients by "
                "capping the global norm of all parameter gradients. "
                "The standard approach clips gradients when the total L2 "
                "norm exceeds a threshold (commonly 1.0 for RNNs, 0.5-5.0 "
                "for transformers). Adaptive gradient clipping (AGC) "
                "proposed by Brock et al. scales each parameter's gradient "
                "based on its own norm relative to the weight norm, which "
                "works better for large models. For mixed precision "
                "training, clip gradients BEFORE the optimizer step but "
                "AFTER unscaling the loss scale. A common pitfall is "
                "clipping too aggressively (threshold < 0.1) which "
                "effectively caps the learning rate and slows convergence "
                "unnecessarily. Monitor the fraction of steps where "
                "clipping is triggered — if >10%, consider increasing the "
                "threshold or investigating the learning rate schedule."
            ),
            metadata={"source_type": "tutorial", "relevance": 0.85},
        ),
    ]


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


def print_corrmap_summary(corrmap: CorrelationMap, profiles: list[PaperProfile]) -> None:
    """Print a compact CorrelationMap summary."""

    print_subheader("CORRELATIONMAP SUMMARY")
    print(f"\n  Papers analyzed:  {corrmap.papers_analyzed}")
    print(f"  Total claims:     {corrmap.total_claims}")
    print(f"  Total components: {corrmap.total_components}")
    print(f"  Total modules:    {corrmap.total_modules}")
    print(f"  Matches found:    {corrmap.matches_found}")
    print(f"    - Component correlations: {len(corrmap.correlations)}")
    print(f"    - Module correlations:    {len(corrmap.module_correlations)}")

    if corrmap.correlations:
        print("\n  Component correlations:")
        for i, c in enumerate(corrmap.correlations, 1):
            print(f"    [{i}] {c.correlation_type} | {truncate(c.paper_title, 45)} "
                  f"-> {c.component.name!r} (score={c.similarity_score:.4f})")

    if corrmap.module_correlations:
        print("\n  Module correlations:")
        for i, m in enumerate(corrmap.module_correlations, 1):
            print(f"    [{i}] {m.correlation_type} | {truncate(m.paper_title, 45)} "
                  f"-> {m.module.module_name!r} (score={m.similarity_score:.4f})")


def print_hypothesis_report(result: HypothesisResult, gen_content_count: int = 0) -> None:
    """Print a comprehensive hypothesis generation report."""

    print_header("HYPOTHESIS GENERATOR E2E REPORT")

    print(f"\n  Input papers:        {result.total_input_papers}")
    print(f"  Input correlations:  {result.total_correlations}")
    print(f"  Gen. content items:  {gen_content_count}")
    print(f"  Reflection rounds:   {result.reflection_rounds_completed}")
    print(f"  Converged:           {result.converged}")
    print(f"  Total hypotheses:    {len(result.hypotheses)}")
    print(f"  Errors:              {len(result.errors)}")

    if result.errors:
        print_subheader("ERRORS (DEGRADED OUTPUT)")
        for err in result.errors:
            print(f"\n  - {err.message}")
            if err.exception_type:
                print(f"    Exception: {err.exception_type}")

    if result.hypotheses:
        ranked = sorted(result.hypotheses, key=lambda h: h.composite, reverse=True)

        print_subheader("GENERATED HYPOTHESES (ranked by composite score)")
        for i, h in enumerate(ranked, 1):
            print(f"\n  [{i}] \"{h.title}\"")
            print(f"      ID:         {h.id}")
            print(f"      Description: {truncate(h.description, 110)}")
            print(f"      Approach:    {truncate(h.approach, 110)}")
            print(f"      Scores:      relevance={h.scores.relevance}  "
                  f"feasibility={h.scores.feasibility}  evidence={h.scores.evidence}")
            print(f"      Composite:   {h.composite:.2f}")
            print(f"      Target:      {h.target_metric}")
            print(f"      Improvement: {truncate(h.expected_improvement, 100)}")
            print(f"      Code changes:{truncate(h.code_changes, 100)}")
            if h.supporting_papers:
                papers_str = ", ".join(truncate(p, 40) for p in h.supporting_papers)
                print(f"      Papers:      {papers_str}")
            if h.correlations:
                print(f"      Correlations:{h.correlations}")
            if h.success_criteria:
                print(f"      Success:     {truncate(h.success_criteria, 100)}")

        print_subheader("REFLECTION LOOP SUMMARY")
        print(f"\n  Rounds completed:    {result.reflection_rounds_completed}")
        print(f"  Converged:           {result.converged}")
        print(f"  Final hypothesis ct: {len(result.hypotheses)}")

        score_distribution = {}
        for h in ranked:
            bucket = f"{int(h.composite)}"
            score_distribution[bucket] = score_distribution.get(bucket, 0) + 1
        print("\n  Composite score distribution:")
        for bucket in sorted(score_distribution, key=int, reverse=True):
            count = score_distribution[bucket]
            bar = "#" * count
            print(f"    {bucket:>2}: {bar} ({count})")

        avg_composite = sum(h.composite for h in ranked) / len(ranked)
        print(f"\n  Average composite:   {avg_composite:.2f}")
        print(f"  Best composite:      {ranked[0].composite:.2f}")
        print(f"  Worst composite:     {ranked[-1].composite:.2f}")
    else:
        print_subheader("NO HYPOTHESES GENERATED")
        print("\n  The pipeline ran but produced no valid hypotheses.")
        print("  This can happen if:")
        print("    - CorrelationMap had no matches (no input to generator)")
        print("    - All generated hypotheses were below the relevance gate")
        print("    - LLM generation failed for all paper-clusters")

    print_header("RESULT")
    if result.hypotheses:
        print(f"\n  SUCCESS: Generated {len(result.hypotheses)} grounded hypotheses "
              f"across {result.total_input_papers} paper-clusters.")
        print(f"  Reflection: {result.reflection_rounds_completed} rounds, "
              f"converged={result.converged}")
    else:
        print(f"\n  PARTIAL/EMPTY: Pipeline completed but no hypotheses survived.")
        if result.errors:
            print(f"  {len(result.errors)} errors were recorded during execution.")
    print(f"{'=' * 72}\n")


# ======================================================================
# Main
# ======================================================================

async def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    QUERY = "How can I improve the training stability and convergence of my ML model?"

    print_header("HYPOTHESIS GENERATOR E2E — Real OpenAI API Calls")
    print(f"\n  Query: \"{QUERY}\"")
    print("\n  Phase 1: Building sample data...")

    profiles = build_paper_profiles()
    codebase = build_codebase_context()
    gen_content = build_general_content()

    print(f"  {len(profiles)} paper profiles loaded")
    print(f"  {len(codebase.components)} components loaded")
    print(f"  {len(codebase.modules)} modules loaded")
    print(f"  {len(gen_content)} general content items loaded")

    # --- Create OpenAI clients ---
    print("\n  Phase 2: Initializing OpenAI clients...")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    embedder = OpenAIEmbedder(client=client, model="text-embedding-3-small")
    classifier = OpenAICorrelationClassifier(client=client, model="gpt-4o-mini")
    corr_config = CorrelationMapConfig(similarity_threshold=0.55)

    # --- Run CorrelationMap pipeline ---
    print("\n  Phase 3: Running CorrelationMap pipeline...")
    print("      -> embed -> cosine -> filter -> classify -> assemble")

    corr_pipeline = CorrelationPipeline(
        embedder=embedder,
        classifier=classifier,
        config=corr_config,
    )

    try:
        corrmap = await corr_pipeline.correlate(
            profiles=profiles,
            codebase=codebase,
        )
        print(f"  CorrelationMap complete: {corrmap.matches_found} matches found")
    except Exception as exc:
        print(f"  ERROR: CorrelationMap failed: {exc}")
        raise

    print_corrmap_summary(corrmap, profiles)

    # --- Run HypothesisGenerator pipeline ---
    print("\n  Phase 4: Running HypothesisGenerator pipeline...")
    print("      -> generate -> reflect -> score -> ground -> rank")

    generator = OpenAIHypothesisGenerator(client=client, model="gpt-4o-mini")
    hypo_config = HypothesisConfig()

    hypo_pipeline = HypothesisPipeline(
        generator=generator,
        embedder=embedder,
        config=hypo_config,
    )

    try:
        result = await hypo_pipeline.run(
            correlation_map=corrmap,
            query=QUERY,
            general_content=gen_content,
        )
        print(f"  Hypothesis pipeline complete: {len(result.hypotheses)} hypotheses")
    except Exception as exc:
        print(f"  ERROR: Hypothesis pipeline failed: {exc}")
        raise

    print_hypothesis_report(result, gen_content_count=len(gen_content))


if __name__ == "__main__":
    asyncio.run(main())
