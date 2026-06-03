"""Unit tests for OpenAICorrelationClassifier adapter.

Uses ``AsyncMock`` to verify API call shapes without real API calls.
No ``OPENAI_API_KEY`` required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.correlation.adapters import OpenAICorrelationClassifier


class TestOpenAICorrelationClassifier:
    """Verify OpenAICorrelationClassifier call shapes and JSON parsing."""

    @pytest.mark.asyncio
    async def test_classify_sends_correct_prompt(self) -> None:
        """classify() sends title + claims + candidates in user message with json_object response_format."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = (
            '{"matches": ['
            '  {"claim_text": "Attention helps accuracy", '
            '   "target_name": "attention_layer", '
            '   "correlation_type": "direct_solution", '
            '   "reasoning": "Direct implementation."}'
            "]}"
        )
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        classifier = OpenAICorrelationClassifier(
            client=mock_client, model="gpt-4o-mini",
        )

        candidates = [
            {
                "claim_text": "Attention helps accuracy",
                "similarity_score": 0.85,
                "target_text": "Attention mechanism implementation",
                "target_type": "component",
                "target_name": "attention_layer",
                "module_name": "model.layers",
            },
        ]

        result = await classifier.classify(
            title="Deep Learning for NLP",
            paper_claims="- Attention helps accuracy\n- Transformers enable parallelization",
            candidates=candidates,
        )

        # Verify single API call was made
        mock_client.chat.completions.create.assert_called_once()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["response_format"] == {"type": "json_object"}

        # Verify messages structure
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Deep Learning for NLP" in messages[1]["content"]
        assert "Attention helps accuracy" in messages[1]["content"]

        assert len(result) == 1
        assert result[0]["correlation_type"] == "direct_solution"
        assert result[0]["reasoning"] == "Direct implementation."

    @pytest.mark.asyncio
    async def test_classify_empty_candidates_returns_empty_list(self) -> None:
        """classify() returns [] when candidates list is empty — no API call."""
        mock_client = AsyncMock()

        classifier = OpenAICorrelationClassifier(client=mock_client)

        result = await classifier.classify(
            title="Test", paper_claims="claims", candidates=[],
        )

        assert result == []
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_classify_handles_json_parse_error(self) -> None:
        """classify() returns [] when LLM returns invalid JSON (CM-05)."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "not valid json{{{{{"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        classifier = OpenAICorrelationClassifier(client=mock_client)

        result = await classifier.classify(
            title="Test",
            paper_claims="- Claim 1",
            candidates=[
                {
                    "claim_text": "Claim 1",
                    "similarity_score": 0.9,
                    "target_text": "Code",
                    "target_type": "component",
                    "target_name": "func",
                    "module_name": "mod",
                },
            ],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_classify_handles_api_exception(self) -> None:
        """classify() returns [] when the API call itself raises (CM-05)."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "API connection error"
        )

        classifier = OpenAICorrelationClassifier(client=mock_client)

        result = await classifier.classify(
            title="Test",
            paper_claims="- Claim 1",
            candidates=[
                {
                    "claim_text": "Claim 1",
                    "similarity_score": 0.9,
                    "target_text": "Code",
                    "target_type": "component",
                    "target_name": "func",
                    "module_name": "mod",
                },
            ],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_classify_handles_none_content(self) -> None:
        """classify() returns [] when LLM content is None (CM-05)."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        classifier = OpenAICorrelationClassifier(client=mock_client)

        result = await classifier.classify(
            title="Test",
            paper_claims="- Claim",
            candidates=[
                {
                    "claim_text": "Claim",
                    "similarity_score": 0.8,
                    "target_text": "Code",
                    "target_type": "component",
                    "target_name": "func",
                    "module_name": "mod",
                },
            ],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_classify_handles_non_dict_response(self) -> None:
        """classify() returns [] when LLM returns a list instead of dict (CM-05)."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '["not", "a", "dict"]'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        classifier = OpenAICorrelationClassifier(client=mock_client)

        result = await classifier.classify(
            title="Test",
            paper_claims="- Claim",
            candidates=[
                {
                    "claim_text": "Claim",
                    "similarity_score": 0.8,
                    "target_text": "Code",
                    "target_type": "component",
                    "target_name": "func",
                    "module_name": "mod",
                },
            ],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_classify_handles_missing_matches_key(self) -> None:
        """classify() returns [] when JSON is valid but lacks 'matches' key."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"other_key": "value"}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        classifier = OpenAICorrelationClassifier(client=mock_client)

        result = await classifier.classify(
            title="Test",
            paper_claims="- Claim",
            candidates=[
                {
                    "claim_text": "Claim",
                    "similarity_score": 0.8,
                    "target_text": "Code",
                    "target_type": "component",
                    "target_name": "func",
                    "module_name": "mod",
                },
            ],
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_classify_handles_non_list_matches(self) -> None:
        """classify() returns [] when 'matches' field is not a list."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"matches": "not_a_list"}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        classifier = OpenAICorrelationClassifier(client=mock_client)

        result = await classifier.classify(
            title="Test",
            paper_claims="- Claim",
            candidates=[
                {
                    "claim_text": "Claim",
                    "similarity_score": 0.8,
                    "target_text": "Code",
                    "target_type": "component",
                    "target_name": "func",
                    "module_name": "mod",
                },
            ],
        )

        assert result == []
