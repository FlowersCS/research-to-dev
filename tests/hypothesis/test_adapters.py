"""Unit tests for OpenAIHypothesisGenerator adapter.

Uses ``AsyncMock`` to verify API call shapes without real API calls.
No ``OPENAI_API_KEY`` required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.hypothesis.adapters import OpenAIHypothesisGenerator


# ===================================================================
# Test data
# ===================================================================


def _sample_correlations() -> list[dict]:
    """Sample correlation dicts as passed by the pipeline."""
    return [
        {
            "id": "a1b2c3d4e5f6",
            "claim_text": "Attention mechanisms improve NLP accuracy",
            "paper_id": "paper-1",
            "paper_title": "Deep Learning for NLP",
            "target_name": "attention_layer",
            "target_type": "component",
            "target_description": "Multi-head attention implementation",
            "correlation_type": "direct_solution",
            "reasoning": "Direct implementation",
            "similarity_score": 0.85,
        },
        {
            "id": "b2c3d4e5f6a7",
            "claim_text": "Transformer architecture enables parallelization",
            "paper_id": "paper-1",
            "paper_title": "Deep Learning for NLP",
            "target_name": "attention_layer",
            "target_type": "component",
            "target_description": "Multi-head attention mechanism",
            "correlation_type": "related_technique",
            "reasoning": "Related technique",
            "similarity_score": 0.72,
        },
    ]


def _sample_hypotheses() -> list[dict]:
    """Sample hypothesis dicts for critique_and_expand."""
    return [
        {
            "id": "hypo123",
            "title": "Improve attention",
            "description": "Replace attention mechanism",
            "approach": "Flash attention",
            "supporting_papers": ["Deep Learning for NLP"],
            "target_metric": "throughput",
            "expected_improvement": "2x",
            "code_changes": "layers.py",
            "scores": {"relevance": 8, "feasibility": 6, "evidence": 7},
            "composite": 7.0,
            "correlations": ["a1b2c3d4e5f6"],
            "success_criteria": "40% improvement",
        },
    ]


def _mock_generate_response() -> AsyncMock:
    """Build an AsyncMock response for generate()."""
    mock_choice = AsyncMock()
    mock_message = AsyncMock()
    mock_message.content = (
        '{"hypotheses": ['
        '  {"title": "Improve attention mechanism", '
        '   "description": "Use flash attention for better throughput", '
        '   "approach": "Replace standard attention with flash attention", '
        '   "supporting_papers": ["Deep Learning for NLP"], '
        '   "target_metric": "training throughput", '
        '   "expected_improvement": "2x faster training", '
        '   "code_changes": "model/layers.py attention_layer", '
        '   "scores": {"relevance": 8, "feasibility": 6, "evidence": 7}, '
        '   "correlations": ["a1b2c3d4e5f6"], '
        '   "success_criteria": "Training time reduced by >40%"}]}'
    )
    mock_choice.message = mock_message
    mock_response = AsyncMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _mock_critique_response() -> AsyncMock:
    """Build an AsyncMock response for critique_and_expand()."""
    mock_choice = AsyncMock()
    mock_message = AsyncMock()
    mock_message.content = (
        '{"critiques": ['
        '  {"hypothesis_title": "Improve attention", '
        '   "weaknesses": ["No empirical evidence cited"]}'
        '], '
        '"new_hypotheses": []}'
    )
    mock_choice.message = mock_message
    mock_response = AsyncMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ===================================================================
# generate() tests
# ===================================================================


class TestOpenAIHypothesisGeneratorGenerate:
    """Verify generate() call shape and response parsing."""

    @pytest.mark.asyncio
    async def test_generate_sends_correct_prompt(self) -> None:
        """generate() sends paper context + correlations + query in user
        message with json_object response_format."""
        mock_client = AsyncMock()
        mock_response = _mock_generate_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(
            client=mock_client, model="gpt-4o-mini",
        )

        correlations = _sample_correlations()
        result = await gen.generate(
            title="Deep Learning for NLP",
            paper_claims="- Attention mechanisms improve NLP accuracy\n"
                         "- Transformer architecture enables parallelization",
            correlations=correlations,
            query="How can I improve my model?",
            round_num=0,
        )

        # Verify call shape
        create_call = mock_client.chat.completions.create
        assert create_call.call_count == 1

        kwargs = create_call.call_args[1]
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["response_format"] == {"type": "json_object"}

        messages = kwargs["messages"]
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert "hypothesis generator" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert "Deep Learning for NLP" in messages[1]["content"]
        assert "How can I improve my model?" in messages[1]["content"]

        # Verify return
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Improve attention mechanism"

    @pytest.mark.asyncio
    async def test_generate_empty_correlations(self) -> None:
        """generate() works with empty correlations list."""
        mock_client = AsyncMock()
        mock_response = _mock_generate_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.generate(
            title="Test Paper",
            paper_claims="- Claim",
            correlations=[],
            query="query",
        )
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_generate_api_exception_returns_empty(self) -> None:
        """generate() returns empty list on API exception (T1)."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("timeout")

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.generate(
            title="Test Paper",
            paper_claims="- Claim",
            correlations=_sample_correlations(),
            query="query",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_null_content_returns_empty(self) -> None:
        """generate() returns empty list on null message content."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.generate(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(), query="q",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_malformed_json_returns_empty(self) -> None:
        """generate() returns empty list on invalid JSON (T4)."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "not valid json{{{"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.generate(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(), query="q",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_non_dict_response_returns_empty(self) -> None:
        """generate() returns empty list when response is not a dict."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "[1, 2, 3]"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.generate(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(), query="q",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_missing_hypotheses_key_returns_empty(self) -> None:
        """generate() returns empty when 'hypotheses' key is missing."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"other_key": "value"}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.generate(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(), query="q",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_hypotheses_not_list_returns_empty(self) -> None:
        """generate() returns empty when 'hypotheses' is not a list."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"hypotheses": "not a list"}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.generate(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(), query="q",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_with_general_content_includes_in_prompt(self) -> None:
        """generate() includes Implementation References section when
        general_content is provided."""
        mock_client = AsyncMock()
        mock_response = _mock_generate_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)

        gen_content = (
            "[1] Stable Training Guide (https://example.com/stable)\n"
            "    Use gradient clipping and layer normalization for stability."
        )

        result = await gen.generate(
            title="Deep Learning for NLP",
            paper_claims="- Attention mechanisms improve NLP accuracy",
            correlations=_sample_correlations(),
            query="How can I improve my model?",
            round_num=0,
            general_content=gen_content,
        )

        # Verify prompt contains the Implementation References section
        create_call = mock_client.chat.completions.create
        messages = create_call.call_args[1]["messages"]
        user_content = messages[1]["content"]
        assert "Implementation References" in user_content
        assert "Stable Training Guide" in user_content
        assert "gradient clipping" in user_content

        # Verify result still produced
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_generate_without_general_content_no_section(self) -> None:
        """generate() does NOT include Implementation References section
        when general_content is omitted (default None)."""
        mock_client = AsyncMock()
        mock_response = _mock_generate_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)

        result = await gen.generate(
            title="Deep Learning for NLP",
            paper_claims="- Attention claim",
            correlations=_sample_correlations(),
            query="How can I improve my model?",
            round_num=0,
        )

        # Verify prompt does NOT contain Implementation References
        create_call = mock_client.chat.completions.create
        messages = create_call.call_args[1]["messages"]
        user_content = messages[1]["content"]
        assert "Implementation References" not in user_content

        # Verify result still produced normally
        assert isinstance(result, list)
        assert len(result) == 1


# ===================================================================
# critique_and_expand() tests
# ===================================================================


class TestOpenAIHypothesisGeneratorCritique:
    """Verify critique_and_expand() call shape and response parsing."""

    @pytest.mark.asyncio
    async def test_critique_sends_correct_prompt(self) -> None:
        """critique_and_expand() sends existing hypotheses + paper
        context in user message with json_object response_format."""
        mock_client = AsyncMock()
        mock_response = _mock_critique_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(
            client=mock_client, model="gpt-4o-mini",
        )

        result = await gen.critique_and_expand(
            title="Deep Learning for NLP",
            paper_claims="- Attention claim",
            correlations=_sample_correlations(),
            existing_hypotheses=_sample_hypotheses(),
            query="How can I improve my model?",
            round_num=1,
        )

        # Verify call shape
        create_call = mock_client.chat.completions.create
        assert create_call.call_count == 1

        kwargs = create_call.call_args[1]
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["response_format"] == {"type": "json_object"}

        messages = kwargs["messages"]
        assert len(messages) == 2
        assert "critic" in messages[0]["content"].lower()
        user_content = messages[1]["content"]
        assert "Deep Learning for NLP" in user_content
        assert "Improve attention" in user_content
        assert "How can I improve my model?" in user_content
        assert "Round: 1" in user_content

        # Verify return
        assert isinstance(result, dict)
        assert "critiques" in result
        assert "new_hypotheses" in result
        assert len(result["critiques"]) == 1
        assert result["critiques"][0]["hypothesis_title"] == "Improve attention"

    @pytest.mark.asyncio
    async def test_critique_api_exception_returns_empty(self) -> None:
        """critique_and_expand() returns empty dict on API exception (T2)."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("timeout")

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.critique_and_expand(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(),
            existing_hypotheses=_sample_hypotheses(),
            query="q", round_num=1,
        )
        assert result == {"critiques": [], "new_hypotheses": []}

    @pytest.mark.asyncio
    async def test_critique_null_content_returns_empty(self) -> None:
        """critique_and_expand() returns empty on null content."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.critique_and_expand(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(),
            existing_hypotheses=_sample_hypotheses(),
            query="q", round_num=1,
        )
        assert result == {"critiques": [], "new_hypotheses": []}

    @pytest.mark.asyncio
    async def test_critique_malformed_json_returns_empty(self) -> None:
        """critique_and_expand() returns empty on invalid JSON (T4)."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "not json"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.critique_and_expand(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(),
            existing_hypotheses=_sample_hypotheses(),
            query="q", round_num=1,
        )
        assert result == {"critiques": [], "new_hypotheses": []}

    @pytest.mark.asyncio
    async def test_critique_non_dict_response_returns_empty(self) -> None:
        """critique_and_expand() returns empty when response is not a dict."""
        mock_client = AsyncMock()
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "[]"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        result = await gen.critique_and_expand(
            title="Test", paper_claims="- C",
            correlations=_sample_correlations(),
            existing_hypotheses=_sample_hypotheses(),
            query="q", round_num=1,
        )
        assert result == {"critiques": [], "new_hypotheses": []}

    @pytest.mark.asyncio
    async def test_critique_default_model_is_gpt4o_mini(self) -> None:
        """Default model is gpt-4o-mini when not specified."""
        mock_client = AsyncMock()
        mock_response = _mock_critique_response()
        mock_client.chat.completions.create.return_value = mock_response

        gen = OpenAIHypothesisGenerator(client=mock_client)
        await gen.critique_and_expand(
            title="T", paper_claims="- C",
            correlations=_sample_correlations(),
            existing_hypotheses=_sample_hypotheses(),
            query="q", round_num=1,
        )

        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs["model"] == "gpt-4o-mini"
