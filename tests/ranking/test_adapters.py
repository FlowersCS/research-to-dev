"""Unit tests for OpenAI adapters (OpenAIEmbedder, OpenAIJudge).

Uses ``AsyncMock`` to verify API call shapes without real API calls.
No ``OPENAI_API_KEY`` required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from research_to_dev.ranking.adapters import OpenAIEmbedder, OpenAIJudge


# ==================================================================
# OpenAIEmbedder
# ==================================================================


class TestOpenAIEmbedder:
    """Verify OpenAIEmbedder call shapes and response handling."""

    @pytest.mark.asyncio
    async def test_embed_calls_api_with_correct_model(self) -> None:
        """embed() sends the configured model and returns embeddings."""
        mock_client = AsyncMock()

        # Mock the embeddings.create() response
        mock_embedding = AsyncMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response = AsyncMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client, model="test-model")

        result = await embedder.embed(["hello world"])

        # Verify API call shape
        mock_client.embeddings.create.assert_called_once_with(
            model="test-model",
            input=["hello world"],
        )

        # Verify return value
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self) -> None:
        """embed() returns one vector per input text in correct order."""
        mock_client = AsyncMock()

        e1 = AsyncMock()
        e1.embedding = [1.0, 0.0]
        e2 = AsyncMock()
        e2.embedding = [0.0, 1.0]
        mock_response = AsyncMock()
        mock_response.data = [e1, e2]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client)

        result = await embedder.embed(["text A", "text B"])

        assert len(result) == 2
        assert result[0] == [1.0, 0.0]
        assert result[1] == [0.0, 1.0]

    @pytest.mark.asyncio
    async def test_embed_query_delegates_to_embed(self) -> None:
        """embed_query() delegates to embed() with a single-element list."""
        mock_client = AsyncMock()

        mock_embedding = AsyncMock()
        mock_embedding.embedding = [0.5, 0.5]
        mock_response = AsyncMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        embedder = OpenAIEmbedder(client=mock_client)

        result = await embedder.embed_query("single query")

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["single query"],
        )
        assert result == [0.5, 0.5]


# ==================================================================
# OpenAIJudge
# ==================================================================


class TestOpenAIJudge:
    """Verify OpenAIJudge batched prompt structure and JSON parsing."""

    @pytest.mark.asyncio
    async def test_judge_sends_batched_prompt(self) -> None:
        """judge() sends all papers in ONE call with json_object response_format."""
        mock_client = AsyncMock()

        # Mock the chat.completions.create() response
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"papers": [{"paper_index": 0, "relevance_score": 0.9, "reasoning": "relevant"}]}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        judge = OpenAIJudge(client=mock_client, model="gpt-4o-mini")

        papers = [
            {
                "paper_index": 0,
                "title": "Deep Learning Advances",
                "abstract": "Recent advances in deep learning.",
                "semantic_score": 0.85,
            },
        ]

        result = await judge.judge("research query", papers)

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
        assert "research query" in messages[1]["content"]

        # Verify parsing
        assert len(result) == 1
        assert result[0]["paper_index"] == 0
        assert result[0]["relevance_score"] == 0.9
        assert result[0]["reasoning"] == "relevant"

    @pytest.mark.asyncio
    async def test_judge_handles_empty_response(self) -> None:
        """judge() returns [] when LLM content is None."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        judge = OpenAIJudge(client=mock_client)

        result = await judge.judge("query", [])

        assert result == []

    @pytest.mark.asyncio
    async def test_judge_handles_json_parse_error(self) -> None:
        """judge() returns [] when LLM returns invalid JSON."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "not valid json!!!"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        judge = OpenAIJudge(client=mock_client)

        result = await judge.judge("query", [])

        assert result == []

    @pytest.mark.asyncio
    async def test_judge_handles_missing_papers_key(self) -> None:
        """judge() returns [] when JSON is valid but lacks 'papers' key."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"other_key": "value"}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        judge = OpenAIJudge(client=mock_client)

        result = await judge.judge("query", [])

        assert result == []
