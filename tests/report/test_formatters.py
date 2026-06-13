"""Tests for format_markdown and format_json — output structure,
content verification, and edge cases per RC-14."""

from __future__ import annotations

import json

from research_to_dev.report.compilation import (
    BaselineComparison,
    CompiledReport,
    HypothesisSummary,
    IterationResult,
    format_json,
    format_markdown,
)
from research_to_dev.report.traceability import (
    CorrelationTrace,
    HypothesisOrigin,
    TraceabilityContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_iterations() -> list[IterationResult]:
    """Return a representative set of iterations."""
    return [
        IterationResult(
            iteration=1,
            metric_name="val_loss",
            metric_value=0.45,
            baseline_value=0.50,
            delta=-0.05,
            status="success",
            timestamp="2026-06-07T14:00:00Z",
        ),
        IterationResult(
            iteration=2,
            metric_name="val_loss",
            metric_value=0.42,
            baseline_value=0.45,
            delta=-0.03,
            status="success",
            timestamp="2026-06-07T14:05:00Z",
        ),
        IterationResult(
            iteration=3,
            metric_name="val_loss",
            metric_value=0.48,
            baseline_value=0.42,
            delta=0.06,
            status="failed",
            timestamp="2026-06-07T14:10:00Z",
        ),
    ]


def _sample_report() -> CompiledReport:
    """Return a CompiledReport with 2 hypotheses."""
    return CompiledReport(
        generated_at="2026-06-07T14:30:00Z",
        hypotheses=[
            HypothesisSummary(
                hypothesis_id="abc123",
                status="improved",
                iterations_total=3,
                iterations_kept=2,
                iterations_discarded=0,
                iterations_crashed=1,
                best_metric_value=0.42,
                baseline_comparison=BaselineComparison(
                    best_improvement=-0.05,
                    overall_trend="improved",
                ),
                iterations=_sample_iterations(),
            ),
            HypothesisSummary(
                hypothesis_id="def456",
                status="crash",
                iterations_total=2,
                iterations_kept=0,
                iterations_discarded=0,
                iterations_crashed=2,
                best_metric_value=None,
                baseline_comparison=None,
                iterations=[
                    IterationResult(
                        iteration=1,
                        metric_name="accuracy",
                        metric_value=0.0,
                        baseline_value=0.90,
                        delta=-0.90,
                        status="failed",
                        timestamp="2026-06-07T14:30:00Z",
                    ),
                    IterationResult(
                        iteration=2,
                        metric_name="accuracy",
                        metric_value=0.0,
                        baseline_value=0.90,
                        delta=-0.90,
                        status="timeout",
                        timestamp="2026-06-07T14:35:00Z",
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# RC-14: format_markdown tests
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    """Markdown formatting structure and content."""

    def test_contains_header_with_timestamp(self) -> None:
        """Markdown output starts with # header + timestamp."""
        report = _sample_report()
        md = format_markdown(report)

        assert md.startswith("# Experiment Results — 2026-06-07T14:30:00Z")

    def test_contains_hypothesis_sections(self) -> None:
        """Each hypothesis gets its own ## section."""
        md = format_markdown(_sample_report())

        assert "## Hypothesis: abc123" in md
        assert "## Hypothesis: def456" in md

    def test_contains_status_field(self) -> None:
        """Status is shown for each hypothesis."""
        md = format_markdown(_sample_report())

        assert "- **Status**: improved" in md
        assert "- **Status**: crash" in md

    def test_contains_iteration_counts(self) -> None:
        """Total/kept/discarded/crashed counts are shown."""
        md = format_markdown(_sample_report())

        assert "3 total" in md
        assert "2 kept" in md
        assert "1 crashed" in md

    def test_contains_best_metric(self) -> None:
        """Best metric value and delta are shown."""
        md = format_markdown(_sample_report())

        assert "val_loss=0.42" in md
        # best_metric_value is 0.42 (from iteration 2), delta is -0.03
        assert "(delta: -0.0300)" in md

    def test_contains_baseline_comparison(self) -> None:
        """Baseline comparison with trend and best improvement."""
        md = format_markdown(_sample_report())

        assert "- **Baseline comparison**: improved (best improvement: -0.0500)" in md

    def test_crash_hypothesis_shows_na(self) -> None:
        """Crash hypothesis shows N/A for metrics."""
        md = format_markdown(_sample_report())

        assert "- **Best metric**: N/A" in md
        assert "N/A (no successful iterations)" in md

    def test_contains_iteration_table(self) -> None:
        """Iteration table with header and data rows."""
        md = format_markdown(_sample_report())

        assert "| # | Metric | Value | Delta | Status |" in md
        assert "| 1 | val_loss | 0.45 | -0.0500 | success |" in md
        assert "| 3 | val_loss | 0.48 | +0.0600 | failed |" in md

    def test_empty_report_no_iterations(self) -> None:
        """Empty report shows appropriate message."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=[],
        )
        md = format_markdown(report)

        assert "*No hypothesis results to report.*" in md

    def test_no_iterations_hypothesis_graceful(self) -> None:
        """Hypothesis with no iterations shows appropriate message."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=[
                HypothesisSummary(
                    hypothesis_id="xyz",
                    status="no_iterations",
                    iterations_total=0,
                    iterations_kept=0,
                    iterations_discarded=0,
                    iterations_crashed=0,
                    best_metric_value=None,
                    baseline_comparison=None,
                    iterations=[],
                ),
            ],
        )
        md = format_markdown(report)

        assert "## Hypothesis: xyz" in md
        assert "- **Status**: no_iterations" in md
        assert "*No iterations recorded.*" in md


# ---------------------------------------------------------------------------
# RC-14: format_json tests
# ---------------------------------------------------------------------------


class TestFormatJson:
    """JSON formatting structure and content."""

    def test_valid_json_output(self) -> None:
        """Output is valid JSON and can be parsed back."""
        report = _sample_report()
        json_str = format_json(report)

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_contains_generated_at(self) -> None:
        """JSON includes generated_at field."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        assert parsed["generated_at"] == "2026-06-07T14:30:00Z"

    def test_contains_hypotheses_array(self) -> None:
        """JSON includes hypotheses array with correct count."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        assert isinstance(parsed["hypotheses"], list)
        assert len(parsed["hypotheses"]) == 2

    def test_hypothesis_fields_present(self) -> None:
        """Each hypothesis has all required fields."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        h = parsed["hypotheses"][0]
        assert "hypothesis_id" in h
        assert "status" in h
        assert "iterations_total" in h
        assert "iterations_kept" in h
        assert "iterations_discarded" in h
        assert "iterations_crashed" in h
        assert "best_metric_value" in h
        assert "baseline_comparison" in h
        assert "iterations" in h

    def test_iteration_fields_present(self) -> None:
        """Each iteration in JSON has all fields."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        it = parsed["hypotheses"][0]["iterations"][0]
        assert "iteration" in it
        assert "metric_name" in it
        assert "metric_value" in it
        assert "baseline_value" in it
        assert "delta" in it
        assert "status" in it
        assert "timestamp" in it

    def test_null_baseline_comparison_serialized(self) -> None:
        """Null baseline_comparison is serialized as null."""
        json_str = format_json(_sample_report())
        parsed = json.loads(json_str)

        crash_h = parsed["hypotheses"][1]  # def456 is crash
        assert crash_h["baseline_comparison"] is None

    def test_json_roundtrip_preserves_data(self) -> None:
        """JSON round-trip preserves all hypothesis data."""
        report = _sample_report()
        json_str = format_json(report)
        parsed = json.loads(json_str)

        h0 = parsed["hypotheses"][0]
        assert h0["hypothesis_id"] == "abc123"
        assert h0["status"] == "improved"
        assert h0["iterations_total"] == 3
        assert h0["iterations_kept"] == 2
        assert h0["best_metric_value"] == 0.42
        assert h0["baseline_comparison"]["overall_trend"] == "improved"
        assert h0["baseline_comparison"]["best_improvement"] == -0.05

    def test_empty_report_json(self) -> None:
        """Empty report produces valid JSON with empty hypotheses."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=[],
        )
        json_str = format_json(report)
        parsed = json.loads(json_str)

        assert parsed["hypotheses"] == []
        assert parsed["generated_at"] == "2026-06-07T14:30:00Z"

    def test_json_indented_output(self) -> None:
        """JSON output is indented for readability."""
        json_str = format_json(_sample_report())
        assert "  " in json_str  # indent=2


# ---------------------------------------------------------------------------
# TR-11: Hypothesis anchors in format_markdown
# ---------------------------------------------------------------------------


class TestFormatMarkdownAnchors:
    """TR-11: Hypothesis heading anchors."""

    def test_anchors_present_without_traceability(self) -> None:
        """Angular anchors render even when traceability is None."""
        report = _sample_report()
        md = format_markdown(report)

        # Anchor tag appears before each hypothesis heading
        assert '<a id="hyp-abc123"></a>' in md
        assert '## Hypothesis: abc123' in md
        assert '<a id="hyp-def456"></a>' in md
        assert '## Hypothesis: def456' in md

    def test_anchors_present_with_traceability(self) -> None:
        """Anchors render when traceability context is provided."""
        report = _sample_report()
        report.traceability = TraceabilityContext(
            hypothesis_origins={},
            claim_anchors={},
            component_anchors={},
        )
        md = format_markdown(report)

        assert '<a id="hyp-abc123"></a>' in md
        assert '<a id="hyp-def456"></a>' in md

    def test_anchor_before_heading(self) -> None:
        """The anchor tag appears immediately before the ## Hypothesis
        heading (same line or preceding line)."""
        report = _sample_report()
        md = format_markdown(report)

        # Verify order: anchor comes before the heading
        anchor_pos = md.find('<a id="hyp-abc123"></a>')
        heading_pos = md.find('## Hypothesis: abc123')
        assert anchor_pos >= 0
        assert heading_pos >= 0
        assert anchor_pos < heading_pos

    def test_empty_report_no_anchors(self) -> None:
        """Empty report with no hypotheses has no anchor tags."""
        report = CompiledReport(
            generated_at="2026-06-07T14:30:00Z",
            hypotheses=[],
        )
        md = format_markdown(report)
        assert '<a id="hyp-' not in md
        assert "*No hypothesis results to report.*" in md


# ---------------------------------------------------------------------------
# TR-12: Traceability subsection in format_markdown
# ---------------------------------------------------------------------------


def _make_trace_context() -> TraceabilityContext:
    """Build a TraceabilityContext with one hypothesis, one resolved and
    one unresolved correlation."""
    return TraceabilityContext(
        hypothesis_origins={
            "abc123": HypothesisOrigin(
                hypothesis_id="abc123",
                title="Optimize gradient descent with Adam",
                description="Replace SGD with Adam optimizer to improve convergence speed and final accuracy on validation set.",
                code_changes="Replace torch.optim.SGD with torch.optim.Adam in train.py",
                supporting_papers=["arxiv-1412.6980", "arxiv-1609.04747"],
                resolved_correlations=[
                    CorrelationTrace(
                        correlation_id="abcd1234abcd",
                        paper_title="Adam: A Method for Stochastic Optimization",
                        claim_text="Adam combines the advantages of AdaGrad and RMSProp.",
                        claim_section="abstract",
                        claim_paper_id="arxiv-1412.6980",
                        component_name="Adam optimizer",
                        component_file_path="/src/train.py",
                        component_module_name="research_to_dev.train",
                        component_signature="def train() -> None:",
                        correlation_type="direct_solution",
                        reasoning="Directly applies Adam as the optimization method.",
                        target_type="component",
                    ),
                ],
                unresolved_correlation_ids=["deadbeefdead"],
            ),
        },
        claim_anchors={
            "claim-abcd1234": "Adam combines the advantages of AdaGrad and RMSProp.",
        },
        component_anchors={
            "comp-research_to_dev-train-abcd1234": (
                "/src/train.py",
                "research_to_dev.train",
                "def train() -> None:",
            ),
        },
    )


class TestFormatMarkdownTraceabilitySubsection:
    """TR-12: Traceability subsections under each hypothesis."""

    def test_subsection_rendered_when_traceability_present(self) -> None:
        """When traceability exists, ### Traceability subsection appears."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert "### Traceability" in md
        # Should appear under hypothesis abc123 (the one with origin data)
        assert "Optimize gradient descent with Adam" in md

    def test_subsection_not_rendered_when_traceability_none(self) -> None:
        """When traceability is None, no ### Traceability subsection."""
        report = _sample_report()
        report.traceability = None
        md = format_markdown(report)

        assert "### Traceability" not in md

    def test_description_truncated(self) -> None:
        """Description longer than 80 chars is truncated with '…'."""
        report = _sample_report()
        ctx = _make_trace_context()
        # Verify the description is longer than 80 chars
        long_desc = ctx.hypothesis_origins["abc123"].description
        assert len(long_desc) > 80
        report.traceability = ctx
        md = format_markdown(report)

        # The truncated version (80 chars) should appear
        truncated = long_desc[:80] + "…"
        assert truncated in md
        # The full version should NOT be in the subsection
        # (the subsection uses truncated, not full)
        # But description may appear in other places, so just verify
        # the truncation prefix is there
        assert long_desc[:80] in md

    def test_supporting_papers_listed(self) -> None:
        """Supporting papers appear in the subsection."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert "arxiv-1412.6980" in md
        assert "arxiv-1609.04747" in md

    def test_resolved_correlations_with_links(self) -> None:
        """Resolved correlations appear with links to appendix."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert "Adam combines" in md
        assert 'comp-' in md or 'claim-' in md  # Links to appendix

    def test_unresolved_correlations_marked(self) -> None:
        """Unresolved correlations are marked with warning."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert "⚠ Unresolved" in md
        assert "deadbeefdead" in md

    def test_appendix_link_present(self) -> None:
        """Link to full traceability appendix is present."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert "[Full traceability detail →]" in md
        assert "(#traceability-appendix)" in md

    def test_empty_correlations_shows_no_resolved(self) -> None:
        """When a hypothesis has no correlations resolved or unresolved,
        show 'No correlations resolved'."""
        report = _sample_report()
        ctx = TraceabilityContext(
            hypothesis_origins={
                "abc123": HypothesisOrigin(
                    hypothesis_id="abc123",
                    title="Simple hypothesis",
                    description="No papers, no correlations.",
                    code_changes="",
                    supporting_papers=[],
                    resolved_correlations=[],
                    unresolved_correlation_ids=[],
                ),
            },
            claim_anchors={},
            component_anchors={},
        )
        report.traceability = ctx
        md = format_markdown(report)

        assert "No correlations resolved" in md
        # Still shows title
        assert "Simple hypothesis" in md


# ---------------------------------------------------------------------------
# TR-13: Traceability appendix in format_markdown
# ---------------------------------------------------------------------------


class TestFormatMarkdownTraceabilityAppendix:
    """TR-13: Traceability appendix after Insights section."""

    def test_appendix_rendered_when_traceability_present(self) -> None:
        """When traceability exists, the appendix section appears."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert "## Traceability Appendix" in md

    def test_appendix_not_rendered_when_traceability_none(self) -> None:
        """When traceability is None, no appendix section appears."""
        report = _sample_report()
        report.traceability = None
        md = format_markdown(report)

        assert "## Traceability Appendix" not in md

    def test_appendix_has_anchor(self) -> None:
        """The appendix section has a traceability-appendix anchor."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert '<a id="traceability-appendix"></a>' in md

    def test_claims_section_with_anchors(self) -> None:
        """Claims section contains claim anchors with hash8 IDs."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert "### Claims" in md
        # The claim anchor should exist
        assert '<a id="claim-' in md
        # The claim text should be present
        assert "Adam combines the advantages of AdaGrad and RMSProp." in md

    def test_code_components_section_with_anchors(self) -> None:
        """Code Components section contains component anchors."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        assert "### Code Components" in md
        assert '<a id="comp-' in md
        assert "research_to_dev.train" in md

    def test_appendix_after_insights(self) -> None:
        """The traceability appendix appears after the Insights section."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        # Appendix should come after any insights content
        # Since no insights are set, appendix should be near the end
        appendix_pos = md.find("## Traceability Appendix")
        assert appendix_pos > 0
        # Should be after the last hypothesis
        last_hyp_pos = md.find("## Hypothesis: def456")
        assert appendix_pos > last_hyp_pos

    def test_cross_links_from_subsection_to_appendix(self) -> None:
        """The subsection links correctly reference appendix anchors."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        md = format_markdown(report)

        # Subsection should have link to appendix
        assert "[Full traceability detail →]" in md
        assert "(#traceability-appendix)" in md

        # Claim anchors should exist
        for anchor_id in report.traceability.claim_anchors:
            assert f'id="{anchor_id}"' in md, f"Claim anchor {anchor_id} not found in output"

    def test_empty_appendix_with_no_correlations(self) -> None:
        """When traceability has no correlations, appendix renders with
        empty sections."""
        report = _sample_report()
        report.traceability = TraceabilityContext(
            hypothesis_origins={
                "abc123": HypothesisOrigin(
                    hypothesis_id="abc123",
                    title="Simple",
                    description="Simple description.",
                    code_changes="",
                ),
            },
            claim_anchors={},
            component_anchors={},
        )
        md = format_markdown(report)

        assert "## Traceability Appendix" in md
        assert "### Claims" in md
        assert "### Code Components" in md


# ---------------------------------------------------------------------------
# TR-14: Traceability key in format_json
# ---------------------------------------------------------------------------


class TestFormatJsonTraceability:
    """TR-14: format_json includes traceability key when present."""

    def test_json_contains_traceability_when_present(self) -> None:
        """When report.traceability exists, JSON has 'traceability' key."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        json_str = format_json(report)
        parsed = json.loads(json_str)

        assert "traceability" in parsed
        trace_data = parsed["traceability"]
        assert "hypothesis_origins" in trace_data
        assert "claim_anchors" in trace_data
        assert "component_anchors" in trace_data

    def test_json_omits_traceability_when_none(self) -> None:
        """When report.traceability is None, JSON has no 'traceability'
        key."""
        report = _sample_report()
        report.traceability = None
        json_str = format_json(report)
        parsed = json.loads(json_str)

        assert "traceability" not in parsed
        assert "generated_at" in parsed
        assert "hypotheses" in parsed

    def test_json_traceability_preserves_hypothesis_origins(self) -> None:
        """The hypothesis_origins data is serialized correctly."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        json_str = format_json(report)
        parsed = json.loads(json_str)

        origins = parsed["traceability"]["hypothesis_origins"]
        assert "abc123" in origins
        origin = origins["abc123"]
        assert origin["title"] == "Optimize gradient descent with Adam"
        assert len(origin["resolved_correlations"]) == 1
        assert len(origin["unresolved_correlation_ids"]) == 1

    def test_json_traceability_preserves_claim_anchors(self) -> None:
        """The claim_anchors mapping is serialized."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        json_str = format_json(report)
        parsed = json.loads(json_str)

        claims = parsed["traceability"]["claim_anchors"]
        # claim_anchors is dict[str, str]
        for anchor_id, claim_text in report.traceability.claim_anchors.items():
            assert claims[anchor_id] == claim_text

    def test_json_traceability_preserves_component_anchors(self) -> None:
        """The component_anchors mapping is serialized."""
        report = _sample_report()
        report.traceability = _make_trace_context()
        json_str = format_json(report)
        parsed = json.loads(json_str)

        comps = parsed["traceability"]["component_anchors"]
        # component_anchors is dict[str, [file_path, module_name, signature]]
        for anchor_id, (fp, mn, sig) in (
            report.traceability.component_anchors.items()
        ):
            assert comps[anchor_id] == [fp, mn, sig]

    def test_json_traceability_roundtrip(self) -> None:
        """JSON round-trip with traceability preserves all data."""
        report = _sample_report()
        ctx = _make_trace_context()
        report.traceability = ctx
        json_str = format_json(report)
        parsed = json.loads(json_str)

        # Verify round-trip preserves hypotheses
        assert len(parsed["hypotheses"]) == 2
        # Verify traceability key
        trace = parsed["traceability"]
        assert len(trace["hypothesis_origins"]) == len(ctx.hypothesis_origins)
        assert len(trace["claim_anchors"]) == len(ctx.claim_anchors)
        assert len(trace["component_anchors"]) == len(ctx.component_anchors)
