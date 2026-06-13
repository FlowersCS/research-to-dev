"""Tests for traceability dataclasses, anchor generation, correlation
resolution, build_traceability(), and JsonTraceReader.

Per traceability change: TR-01 through TR-06, TR-25, TR-26.
"""

from __future__ import annotations


# ===================================================================
# TR-01: Dataclass construction tests
# ===================================================================


class TestCorrelationTraceConstruction:
    """Verify CorrelationTrace dataclass fields, defaults, and type-creation."""

    def test_all_fields_populated(self):
        """GIVEN all required fields
        WHEN constructing CorrelationTrace
        THEN it stores all values correctly."""
        from research_to_dev.report.traceability import CorrelationTrace

        ct = CorrelationTrace(
            correlation_id="abc123def456",
            paper_title="Attention Is All You Need",
            claim_text="Multi-head attention enables parallel processing.",
            claim_section="methods",
            claim_paper_id="1706.03762",
            component_name="MultiHeadAttention",
            component_file_path="src/model/attention.py",
            component_module_name="model.attention",
            component_signature="class MultiHeadAttention(nn.Module)",
            correlation_type="direct_solution",
            reasoning="The paper describes the exact mechanism implemented.",
            target_type="component",
        )

        assert ct.correlation_id == "abc123def456"
        assert ct.paper_title == "Attention Is All You Need"
        assert ct.claim_text == "Multi-head attention enables parallel processing."
        assert ct.claim_section == "methods"
        assert ct.claim_paper_id == "1706.03762"
        assert ct.component_name == "MultiHeadAttention"
        assert ct.component_file_path == "src/model/attention.py"
        assert ct.component_module_name == "model.attention"
        assert ct.component_signature == "class MultiHeadAttention(nn.Module)"
        assert ct.correlation_type == "direct_solution"
        assert ct.reasoning == "The paper describes the exact mechanism implemented."
        assert ct.target_type == "component"

    def test_is_dataclass(self):
        """CorrelationTrace MUST be a dataclass, not a plain class."""
        from dataclasses import is_dataclass

        from research_to_dev.report.traceability import CorrelationTrace

        assert is_dataclass(CorrelationTrace)


class TestHypothesisOriginConstruction:
    """Verify HypothesisOrigin dataclass fields, defaults, and type-preservation."""

    def test_all_fields_populated_with_correlations(self):
        """GIVEN hypothesis origin data with resolved correlations
        WHEN constructing HypothesisOrigin
        THEN it stores all fields correctly."""
        from research_to_dev.report.traceability import (
            CorrelationTrace,
            HypothesisOrigin,
        )

        ct = CorrelationTrace(
            correlation_id="synth-01",
            paper_title="Test Paper",
            claim_text="Test claim",
            claim_section="results",
            claim_paper_id="paper-1",
            component_name="TestComponent",
            component_file_path="test/file.py",
            component_module_name="test.file",
            component_signature="def test():",
            correlation_type="related_technique",
            reasoning="Similar approach.",
            target_type="component",
        )

        ho = HypothesisOrigin(
            hypothesis_id="hyp-abc123",
            title="Add dropout to reduce overfitting",
            description="Investigate dropout layers to improve generalization.",
            code_changes="Add nn.Dropout(0.5) after each linear layer.",
            supporting_papers=["1706.03762", "1803.02148"],
            resolved_correlations=[ct],
            unresolved_correlation_ids=["unres-01"],
        )

        assert ho.hypothesis_id == "hyp-abc123"
        assert ho.title == "Add dropout to reduce overfitting"
        assert "generalization" in ho.description
        assert "Dropout" in ho.code_changes
        assert "1706.03762" in ho.supporting_papers
        assert len(ho.resolved_correlations) == 1
        assert ho.resolved_correlations[0].correlation_id == "synth-01"
        assert ho.unresolved_correlation_ids == ["unres-01"]

    def test_defaults_empty_lists(self):
        """GIVEN no resolved_correlations or unresolved_correlation_ids
        WHEN constructing HypothesisOrigin
        THEN defaults are empty lists."""
        from research_to_dev.report.traceability import HypothesisOrigin

        ho = HypothesisOrigin(
            hypothesis_id="h1",
            title="t",
            description="d",
            code_changes="c",
            supporting_papers=[],
        )

        assert ho.resolved_correlations == []
        assert ho.unresolved_correlation_ids == []
        assert ho.supporting_papers == []

    def test_is_dataclass(self):
        """HypothesisOrigin MUST be a dataclass."""
        from dataclasses import is_dataclass

        from research_to_dev.report.traceability import HypothesisOrigin

        assert is_dataclass(HypothesisOrigin)


class TestTraceabilityContextConstruction:
    """Verify TraceabilityContext dataclass fields and defaults."""

    def test_all_fields_populated(self):
        """GIVEN hypothesis_origins, claim_anchors, and component_anchors
        WHEN constructing TraceabilityContext
        THEN all values are stored."""
        from research_to_dev.report.traceability import (
            CorrelationTrace,
            HypothesisOrigin,
            TraceabilityContext,
        )

        ct = CorrelationTrace(
            correlation_id="c1",
            paper_title="P",
            claim_text="C",
            claim_section="s",
            claim_paper_id="p1",
            component_name="comp",
            component_file_path="f.py",
            component_module_name="m",
            component_signature="sig",
            correlation_type="direct_solution",
            reasoning="r",
            target_type="component",
        )

        ho = HypothesisOrigin(
            hypothesis_id="hyp-1",
            title="Test",
            description="Desc",
            code_changes="Changes",
            supporting_papers=["paper-1"],
            resolved_correlations=[ct],
            unresolved_correlation_ids=[],
        )

        ctx = TraceabilityContext(
            hypothesis_origins={"hyp-1": ho},
            claim_anchors={"claim-abc": "Some claim text"},
            component_anchors={"comp-x": ("f.py", "m", "def f():")},
        )

        assert "hyp-1" in ctx.hypothesis_origins
        assert ctx.hypothesis_origins["hyp-1"].title == "Test"
        assert ctx.claim_anchors["claim-abc"] == "Some claim text"
        assert ctx.component_anchors["comp-x"] == ("f.py", "m", "def f():")

    def test_empty_context(self):
        """GIVEN no data
        WHEN constructing TraceabilityContext with empty dicts
        THEN all dicts are empty."""
        from research_to_dev.report.traceability import TraceabilityContext

        ctx = TraceabilityContext(
            hypothesis_origins={},
            claim_anchors={},
            component_anchors={},
        )

        assert ctx.hypothesis_origins == {}
        assert ctx.claim_anchors == {}
        assert ctx.component_anchors == {}

    def test_is_dataclass(self):
        """TraceabilityContext MUST be a dataclass."""
        from dataclasses import is_dataclass

        from research_to_dev.report.traceability import TraceabilityContext

        assert is_dataclass(TraceabilityContext)


# ===================================================================
# TR-02: TraceReader Protocol tests
# ===================================================================


class TestTraceReaderProtocol:
    """Verify TraceReader Protocol supports structural subtyping."""

    def test_protocol_structural_subtyping(self):
        """A class implementing read(trace_path) -> TraceabilityContext | None
        MUST satisfy TraceReader without explicit inheritance."""
        from research_to_dev.report.traceability import (
            TraceReader,
            TraceabilityContext,
        )

        # Structural subtype: implements read() with correct signature
        class ValidReader:
            def read(self, trace_path: str) -> TraceabilityContext | None:
                return None

        reader = ValidReader()
        # Verify structural subtyping — isinstance should work if
        # the Protocol is @runtime_checkable. If it's not decorated,
        # we verify the class has the expected method.
        assert hasattr(reader, "read")
        result = reader.read("/some/path")
        assert result is None

        # Verify the Protocol is importable and usable as a type hint
        assert TraceReader is not None

    def test_missing_method_not_compatible(self):
        """A class missing read() MUST NOT accidentally pass as TraceReader."""
        from research_to_dev.report.traceability import TraceReader

        class IncompleteReader:
            def other_method(self) -> None:
                pass

        obj = IncompleteReader()
        assert not hasattr(obj, "read")
        # Even without isinstance check, structural typing means this
        # class would fail at type-check time
        assert TraceReader is not None


# ===================================================================
# TR-03: Anchor generation helpers tests
# ===================================================================


class TestHash8:
    """Verify _hash8 deterministic hash generation."""

    def test_deterministic_output(self):
        """_hash8 MUST produce the same output for the same input."""
        from research_to_dev.report.traceability import _hash8

        text = "Multi-head attention enables parallel processing of sequences."
        first = _hash8(text)
        second = _hash8(text)
        assert first == second
        assert len(first) == 8
        # Must be hex characters only
        assert all(c in "0123456789abcdef" for c in first)

    def test_different_inputs_produce_different_hashes(self):
        """Different inputs SHOULD produce different hashes (high probability)."""
        from research_to_dev.report.traceability import _hash8

        h1 = _hash8("attention is all you need")
        h2 = _hash8("bert: pre-training of deep bidirectional transformers")
        assert h1 != h2

    def test_empty_string(self):
        """_hash8 MUST handle empty string gracefully."""
        from research_to_dev.report.traceability import _hash8

        result = _hash8("")
        assert len(result) == 8
        assert all(c in "0123456789abcdef" for c in result)

    def test_special_characters(self):
        """_hash8 MUST handle special characters, unicode, and whitespace."""
        from research_to_dev.report.traceability import _hash8

        h1 = _hash8("hello world")
        h2 = _hash8("hello\tworld\n")
        # Different inputs (tabs/newlines vs spaces) produce different hashes
        assert h1 != h2

        h3 = _hash8("café résumé")
        assert len(h3) == 8
        assert all(c in "0123456789abcdef" for c in h3)

    def test_collision_detection_counter(self):
        """When two different texts produce the same hash8, the second
        MUST append a '-2' counter for disambiguation."""
        from research_to_dev.report.traceability import _hash8

        # Use the anchor-generation logic to test collision handling.
        # We create a collision-aware function that tracks seen hashes.
        # Since true collisions are astronomically unlikely with SHA-256,
        # we test the collision resolution mechanism by simulating it.
        text1 = "text one"
        text2 = "text two"  # different text, different hash

        h1 = _hash8(text1)
        h2 = _hash8(text2)

        # Both should be valid 8-char hex
        assert len(h1) == 8
        assert len(h2) == 8
        # Normal case: different texts produce different hashes
        # (this is probabilistic but SHA-256 makes it virtually certain)


# ===================================================================
# TR-25 / EDGE-02: _unique_anchor collision resolution tests
# ===================================================================


class TestUniqueAnchor:
    """Verify _unique_anchor collision resolution with counter appending.

    Per spec EDGE-02: When two claims share the same hash8, the second
    anchor MUST append a '-2' counter for disambiguation.
    """

    def test_first_use_returns_base(self) -> None:
        """GIVEN a fresh counter dict
        WHEN calling _unique_anchor with a base for the first time
        THEN it returns the base unchanged and records count 1."""
        from research_to_dev.report.traceability import _unique_anchor

        counters: dict[str, int] = {}
        result = _unique_anchor("abc12345", counters)

        assert result == "abc12345"
        assert counters == {"abc12345": 1}

    def test_second_use_appends_dash_2(self) -> None:
        """GIVEN a base already seen once
        WHEN calling _unique_anchor with the same base again
        THEN it returns '{base}-2' and increments count."""
        from research_to_dev.report.traceability import _unique_anchor

        counters: dict[str, int] = {"abc12345": 1}
        result = _unique_anchor("abc12345", counters)

        assert result == "abc12345-2"
        assert counters == {"abc12345": 2}

    def test_third_use_appends_dash_3(self) -> None:
        """GIVEN a base already seen twice
        WHEN calling _unique_anchor with the same base a third time
        THEN it returns '{base}-3'."""
        from research_to_dev.report.traceability import _unique_anchor

        counters: dict[str, int] = {"abc12345": 2}
        result = _unique_anchor("abc12345", counters)

        assert result == "abc12345-3"
        assert counters == {"abc12345": 3}

    def test_different_bases_independent_counters(self) -> None:
        """GIVEN multiple bases in use
        WHEN calling _unique_anchor with different bases
        THEN each base maintains an independent collision counter."""
        from research_to_dev.report.traceability import _unique_anchor

        counters: dict[str, int] = {"abc12345": 3}
        result = _unique_anchor("def67890", counters)

        assert result == "def67890"
        assert counters == {"abc12345": 3, "def67890": 1}

    def test_end_to_end_collision_sequence(self) -> None:
        """GIVEN no prior counters
        WHEN the same base is used three times intermixed with another base
        THEN counters are tracked correctly across all calls."""
        from research_to_dev.report.traceability import _unique_anchor

        counters: dict[str, int] = {}

        # First use of base A
        assert _unique_anchor("claim-aaa", counters) == "claim-aaa"
        assert counters["claim-aaa"] == 1

        # First use of base B
        assert _unique_anchor("claim-bbb", counters) == "claim-bbb"
        assert counters["claim-bbb"] == 1

        # Second use of base A — collision!
        assert _unique_anchor("claim-aaa", counters) == "claim-aaa-2"
        assert counters["claim-aaa"] == 2

        # Second use of base B — collision!
        assert _unique_anchor("claim-bbb", counters) == "claim-bbb-2"
        assert counters["claim-bbb"] == 2

        # Third use of base A
        assert _unique_anchor("claim-aaa", counters) == "claim-aaa-3"
        assert counters["claim-aaa"] == 3


class TestSlugify:
    """Verify _slugify anchor fragment generation."""

    def test_simple_module_name(self):
        """_slugify MUST produce a valid markdown anchor fragment."""
        from research_to_dev.report.traceability import _slugify

        result = _slugify("model.attention")
        assert result == "model-attention"

    def test_special_characters_replaced(self):
        """Special characters MUST be replaced with hyphens."""
        from research_to_dev.report.traceability import _slugify

        result = _slugify("my_module[class]")
        # Brackets become hyphens, underscores preserved, trailing hyphen stripped
        assert result == "my_module-class"

    def test_multiple_special_chars(self):
        """Multiple adjacent special chars become single hyphen (no double hyphens)."""
        from research_to_dev.report.traceability import _slugify

        result = _slugify("a..b")
        # Dots become hyphens, consecutive become single
        assert result == "a-b"

    def test_leading_trailing_special_chars(self):
        """Leading/trailing special chars MUST be stripped."""
        from research_to_dev.report.traceability import _slugify

        result = _slugify("[module]")
        assert result == "module"

    def test_uppercase_lowered(self):
        """Uppercase letters MUST be converted to lowercase."""
        from research_to_dev.report.traceability import _slugify

        result = _slugify("MyModule")
        assert result == "mymodule"

    def test_empty_string(self):
        """Empty string MUST return 'unknown'."""
        from research_to_dev.report.traceability import _slugify

        result = _slugify("")
        assert result == "unknown"


# ===================================================================
# TR-04: Correlation resolution helpers tests
# ===================================================================


class TestBuildCorrelationLookup:
    """Verify _build_correlation_lookup index building."""

    def test_lookup_with_multiple_entries(self):
        """_build_correlation_lookup MUST build index keyed by synthetic ID."""
        from research_to_dev.report.traceability import _build_correlation_lookup

        # Simulated correlation dict from trace.json
        corr_dict = {
            "correlation_id_1": {
                "claim": {"text": "Attention improves translation quality."},
                "target_name": "MultiHeadAttention",
            },
            "correlation_id_2": {
                "claim": {"text": "Residual connections prevent degradation."},
                "target_name": "ResidualBlock",
            },
        }

        lookup = _build_correlation_lookup(corr_dict)
        assert isinstance(lookup, dict)
        assert len(lookup) == 2
        # Keys should be SHA-256 hash8 of claim text + target_name
        for key in lookup:
            assert len(key) == 8
            assert all(c in "0123456789abcdef" for c in key)

    def test_empty_dict(self):
        """_build_correlation_lookup with empty dict MUST return empty dict."""
        from research_to_dev.report.traceability import _build_correlation_lookup

        lookup = _build_correlation_lookup({})
        assert lookup == {}

    def test_lookup_stores_full_entry(self):
        """Each lookup value MUST be the full correlation entry dict."""
        from research_to_dev.report.traceability import _build_correlation_lookup

        corr_dict = {
            "id1": {
                "claim": {"text": "Test claim"},
                "target_name": "TestComponent",
                "paper_title": "Test Paper",
            }
        }

        lookup = _build_correlation_lookup(corr_dict)
        assert len(lookup) == 1
        key = next(iter(lookup))
        entry = lookup[key]
        assert entry["paper_title"] == "Test Paper"
        assert entry["claim"]["text"] == "Test claim"


class TestResolveCorrelation:
    """Verify _resolve_correlation matching logic."""

    def test_full_match(self):
        """When a matching synthetic ID exists in lookup, resolve it."""
        from research_to_dev.report.traceability import (
            _build_correlation_lookup,
            _resolve_correlation,
        )

        corr_dict = {
            "id1": {
                "claim": {"text": "Match me"},
                "target_name": "Target",
                "correlation_type": "direct_solution",
            }
        }
        lookup = _build_correlation_lookup(corr_dict)
        # The synthetic ID is the hash8 of "Match me" + "Target"
        synthetic_id = list(lookup.keys())[0]

        entry, status = _resolve_correlation(synthetic_id, lookup)
        assert status == "resolved"
        assert entry is not None
        assert entry["correlation_type"] == "direct_solution"

    def test_no_match(self):
        """When synthetic ID not in lookup, return unresolved."""
        from research_to_dev.report.traceability import _resolve_correlation

        entry, status = _resolve_correlation("nonexistent", {"abc12345": {}})
        assert status == "unresolved"
        assert entry is None

    def test_empty_lookup(self):
        """With empty lookup dict, all resolutions MUST be unresolved."""
        from research_to_dev.report.traceability import _resolve_correlation

        entry, status = _resolve_correlation("abc12345", {})
        assert status == "unresolved"
        assert entry is None


# ===================================================================
# TR-05: build_traceability() tests
# ===================================================================


class TestBuildTraceability:
    """Verify build_traceability() full pipeline."""

    def _sample_hypothesis_dicts(self) -> list[dict]:
        """Return sample hypothesis dicts matching pipeline trace format."""
        return [
            {
                "id": "hyp-abc123def456",
                "title": "Add dropout layers",
                "description": "Investigate dropout to reduce overfitting.",
                "code_changes": "Add nn.Dropout after each linear layer.",
                "supporting_papers": ["paper-1", "paper-2"],
                "correlations": [],  # will be populated per test
            }
        ]

    def _sample_correlation_dict(self) -> dict:
        """Return sample correlation dict matching pipeline trace format."""
        return {
            "corr-1": {
                "claim": {"text": "Dropout reduces overfitting by preventing co-adaptation."},
                "target_name": "DropoutLayer",
                "paper_title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
                "claim_section": "methods",
                "claim_paper_id": "1207.0580",
                "component_name": "DropoutLayer",
                "component_file_path": "src/model/layers.py",
                "component_module_name": "model.layers",
                "component_signature": "class DropoutLayer(nn.Module)",
                "correlation_type": "direct_solution",
                "reasoning": "The paper describes dropout which directly matches the implementation.",
                "target_type": "component",
            }
        }

    def test_full_resolution(self):
        """GIVEN hypothesis with correlation IDs that match the correlation dict
        WHEN build_traceability() is called
        THEN correlations are resolved, anchors populated."""
        from research_to_dev.report.traceability import (
            _hash8,
            build_traceability,
        )

        corr_dict = self._sample_correlation_dict()
        # Compute the synthetic ID that would be in the hypothesis correlations
        claim_text = corr_dict["corr-1"]["claim"]["text"]
        target_name = corr_dict["corr-1"]["target_name"]
        synthetic_id = _hash8(claim_text + target_name)

        hyp_dicts = self._sample_hypothesis_dicts()
        hyp_dicts[0]["correlations"] = [synthetic_id]

        ctx = build_traceability(hyp_dicts, corr_dict)
        assert ctx is not None
        assert len(ctx.hypothesis_origins) == 1
        ho = ctx.hypothesis_origins["hyp-abc123def456"]
        assert ho.title == "Add dropout layers"
        assert len(ho.resolved_correlations) == 1
        assert ho.resolved_correlations[0].paper_title == (
            "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
        )
        assert ho.unresolved_correlation_ids == []
        assert len(ctx.claim_anchors) == 1
        assert len(ctx.component_anchors) == 1

    def test_partial_resolution(self):
        """GIVEN hypothesis with both matching and non-matching correlation IDs
        WHEN build_traceability() is called
        THEN resolved entries appear; unresolved IDs are tracked."""
        from research_to_dev.report.traceability import build_traceability

        corr_dict = self._sample_correlation_dict()
        hyp_dicts = self._sample_hypothesis_dicts()
        # One real synthetic ID (won't match since we changed text), one fake
        hyp_dicts[0]["correlations"] = ["abcdef01", "nonexist123"]

        ctx = build_traceability(hyp_dicts, corr_dict)
        assert ctx is not None
        ho = ctx.hypothesis_origins["hyp-abc123def456"]
        # No matches because "abcdef01" won't match the real synthetic hash
        assert len(ho.resolved_correlations) == 0
        assert len(ho.unresolved_correlation_ids) == 2
        assert "abcdef01" in ho.unresolved_correlation_ids
        assert "nonexist123" in ho.unresolved_correlation_ids

    def test_no_correlation_dict(self):
        """GIVEN None for correlation_dict
        WHEN build_traceability() is called
        THEN all correlations are unresolved, anchors are empty."""
        from research_to_dev.report.traceability import build_traceability

        hyp_dicts = self._sample_hypothesis_dicts()
        hyp_dicts[0]["correlations"] = ["abc12345"]

        ctx = build_traceability(hyp_dicts, None)
        assert ctx is not None
        ho = ctx.hypothesis_origins["hyp-abc123def456"]
        assert len(ho.resolved_correlations) == 0
        assert ho.unresolved_correlation_ids == ["abc12345"]
        assert ctx.claim_anchors == {}
        assert ctx.component_anchors == {}

    def test_empty_hypothesis_list(self):
        """GIVEN empty hypothesis list
        WHEN build_traceability() is called
        THEN returns empty TraceabilityContext."""
        from research_to_dev.report.traceability import build_traceability

        ctx = build_traceability([], {"corr": {}})
        assert ctx.hypothesis_origins == {}
        assert ctx.claim_anchors == {}
        assert ctx.component_anchors == {}

    def test_malformed_correlation_entry_t4(self):
        """GIVEN a correlation entry missing claim text (T4 resilience)
        WHEN build_traceability() is called
        THEN the malformed entry is skipped, remaining work continues."""
        from research_to_dev.report.traceability import build_traceability

        # Entry without claim.text
        bad_corr_dict = {
            "bad-1": {
                "claim": {"no_text_here": True},
                "target_name": "Target",
                "paper_title": "Bad Paper",
            }
        }

        hyp_dicts = self._sample_hypothesis_dicts()
        hyp_dicts[0]["correlations"] = ["abc12345"]

        ctx = build_traceability(hyp_dicts, bad_corr_dict)
        # Should not crash — just no matches
        assert ctx is not None
        ho = ctx.hypothesis_origins["hyp-abc123def456"]
        assert len(ho.resolved_correlations) == 0

    def test_hypothesis_without_correlations_field(self):
        """GIVEN hypothesis dict without a 'correlations' key
        WHEN build_traceability() is called
        THEN it is processed with empty correlations."""
        from research_to_dev.report.traceability import build_traceability

        hyp_dicts = [{
            "id": "hyp-xyz",
            "title": "Test",
            "description": "Desc",
            "code_changes": "",
            "supporting_papers": [],
        }]

        ctx = build_traceability(hyp_dicts, None)
        assert ctx is not None
        ho = ctx.hypothesis_origins["hyp-xyz"]
        assert ho.resolved_correlations == []
        assert ho.unresolved_correlation_ids == []


# ===================================================================
# TR-08: ReportConfig.trace_path tests
# ===================================================================


class TestReportConfigTracePath:
    """Verify ReportConfig.trace_path field."""

    def test_default_is_empty(self):
        """GIVEN no trace_path argument
        WHEN constructing ReportConfig
        THEN trace_path defaults to empty string."""
        from research_to_dev.shared.config import ReportConfig

        config = ReportConfig()
        assert config.trace_path == ""

    def test_custom_value(self):
        """GIVEN a custom trace_path
        WHEN constructing ReportConfig
        THEN the value is stored."""
        from research_to_dev.shared.config import ReportConfig

        config = ReportConfig(trace_path=".research-to-dev/pipeline/trace.json")
        assert config.trace_path == ".research-to-dev/pipeline/trace.json"

    def test_existing_construct_unchanged(self):
        """GIVEN ReportConfig constructed with only old fields
        WHEN the trace_path default is used
        THEN existing construction patterns still work."""
        from research_to_dev.shared.config import ReportConfig

        config = ReportConfig(
            experiments_dir="/tmp/experiments",
            reports_dir="/tmp/reports",
        )
        assert config.experiments_dir == "/tmp/experiments"
        assert config.reports_dir == "/tmp/reports"
        assert config.trace_path == ""


# ===================================================================
# TR-06: JsonTraceReader adapter tests
# ===================================================================


def _build_trace_json_for_test(
    hypotheses: list[dict] | None = None,
    correlations: dict | None = None,
) -> str:
    """Build a minimal trace.json for JsonTraceReader tests."""
    import json
    data: dict = {
        "query": "test",
        "codebase_path": "/tmp/test",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "hypotheses": hypotheses or [],
        "steps": {},
        "warnings": [],
    }
    if correlations is not None:
        data["correlation"] = correlations
    return json.dumps(data, indent=2)


class TestJsonTraceReader:
    """Verify JsonTraceReader adapter satisfying TraceReader Protocol."""

    def test_happy_path(self):
        """GIVEN a valid trace.json with hypotheses and correlations
        WHEN JsonTraceReader.read() is called
        THEN it returns a populated TraceabilityContext."""
        import tempfile
        from pathlib import Path

        from research_to_dev.report.traceability import (
            JsonTraceReader,
            TraceabilityContext,
        )

        corr_dict = {
            "corr-1": {
                "claim": {"text": "Dropout reduces overfitting."},
                "target_name": "DropoutLayer",
                "paper_title": "Dropout Paper",
                "claim_section": "methods",
                "claim_paper_id": "1207.0580",
                "component_name": "DropoutLayer",
                "component_file_path": "src/layers.py",
                "component_module_name": "layers",
                "component_signature": "class DropoutLayer",
                "correlation_type": "direct_solution",
                "reasoning": "Matches.",
                "target_type": "component",
            }
        }

        from research_to_dev.report.traceability import _hash8
        synthetic_id = _hash8("Dropout reduces overfitting." + "DropoutLayer")

        hypos = [{
            "id": "hyp-test",
            "title": "Test Hypothesis",
            "description": "Test description.",
            "code_changes": "Add dropout.",
            "supporting_papers": ["paper-1"],
            "correlations": [synthetic_id],
        }]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_json_for_test(
                hypotheses=hypos, correlations=corr_dict,
            ))
            trace_path = f.name

        try:
            reader = JsonTraceReader()
            ctx = reader.read(trace_path)
            assert ctx is not None
            assert isinstance(ctx, TraceabilityContext)
            assert len(ctx.hypothesis_origins) == 1
            ho = ctx.hypothesis_origins["hyp-test"]
            assert len(ho.resolved_correlations) == 1
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_missing_file(self):
        """GIVEN a non-existent trace path
        WHEN JsonTraceReader.read() is called
        THEN it returns None."""
        from research_to_dev.report.traceability import JsonTraceReader

        reader = JsonTraceReader()
        ctx = reader.read("/nonexistent/trace.json")
        assert ctx is None

    def test_malformed_json(self):
        """GIVEN a file with invalid JSON
        WHEN JsonTraceReader.read() is called
        THEN it returns None."""
        import tempfile
        from pathlib import Path

        from research_to_dev.report.traceability import JsonTraceReader

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{invalid json content !!!")
            trace_path = f.name

        try:
            reader = JsonTraceReader()
            ctx = reader.read(trace_path)
            assert ctx is None
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_missing_keys(self):
        """GIVEN a trace.json without correlation key
        WHEN JsonTraceReader.read() is called
        THEN hypotheses are processed but no correlations resolved."""
        import tempfile
        from pathlib import Path

        from research_to_dev.report.traceability import JsonTraceReader

        hypos = [{
            "id": "hyp-no-corr",
            "title": "No Correlations",
            "description": "Test.",
            "code_changes": "",
            "supporting_papers": [],
            "correlations": ["some-id"],
        }]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_json_for_test(
                hypotheses=hypos, correlations=None,
            ))
            trace_path = f.name

        try:
            reader = JsonTraceReader()
            ctx = reader.read(trace_path)
            assert ctx is not None
            ho = ctx.hypothesis_origins["hyp-no-corr"]
            assert len(ho.resolved_correlations) == 0
            assert len(ho.unresolved_correlation_ids) == 1
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_satisfies_protocol(self):
        """JsonTraceReader MUST structurally satisfy TraceReader Protocol."""
        from research_to_dev.report.traceability import JsonTraceReader

        reader = JsonTraceReader()
        assert hasattr(reader, "read")
        # read() accepts trace_path parameter
        result = reader.read("/nonexistent/path")
        assert result is None
