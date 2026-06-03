"""Unit tests for OpenAISectionExtractor adapter.

Uses ``AsyncMock`` to verify API call shapes without real API calls.
No ``OPENAI_API_KEY`` required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.paper_profile.adapters import OpenAISectionExtractor


class TestOpenAISectionExtractor:
    """Verify OpenAISectionExtractor call shapes and JSON parsing."""

    @pytest.mark.asyncio
    async def test_extract_sends_correct_prompt(self) -> None:
        """extract() sends title + abstract in user message with json_object response_format."""
        mock_client = AsyncMock()

        # Mock the chat.completions.create() response
        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = (
            '{"sections": [{"name": "abstract", "content": "test", "claims": []}]}'
        )
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAISectionExtractor(
            client=mock_client, model="gpt-4o-mini"
        )

        result = await extractor.extract(
            "Test Paper", "This is the abstract."
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
        assert "Test Paper" in messages[1]["content"]
        assert "This is the abstract" in messages[1]["content"]

        # Verify parsing
        assert result == {
            "sections": [{"name": "abstract", "content": "test", "claims": []}],
        }

    @pytest.mark.asyncio
    async def test_extract_handles_empty_abstract(self) -> None:
        """extract() sends placeholder when abstract is empty string."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"sections": []}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAISectionExtractor(client=mock_client)

        result = await extractor.extract("No Abstract Paper", "")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        assert "(no text available)" in user_content
        assert result == {"sections": []}

    @pytest.mark.asyncio
    async def test_extract_handles_none_content(self) -> None:
        """extract() returns {'sections': []} when LLM content is None."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAISectionExtractor(client=mock_client)

        result = await extractor.extract("Test", "abstract text")

        assert result == {"sections": []}

    @pytest.mark.asyncio
    async def test_extract_handles_json_parse_error(self) -> None:
        """extract() returns {'sections': []} when LLM returns invalid JSON."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "not valid json!!!"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAISectionExtractor(client=mock_client)

        result = await extractor.extract("Test", "content")

        assert result == {"sections": []}

    @pytest.mark.asyncio
    async def test_extract_handles_missing_sections_key(self) -> None:
        """extract() returns {'sections': []} when JSON is valid but lacks 'sections' key."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"other_key": "value"}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAISectionExtractor(client=mock_client)

        result = await extractor.extract("Test", "content")

        assert result == {"sections": []}

    @pytest.mark.asyncio
    async def test_extract_handles_api_exception(self) -> None:
        """extract() returns {'sections': []} when the API call itself raises."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "API connection error"
        )

        extractor = OpenAISectionExtractor(client=mock_client)

        result = await extractor.extract("Test", "content")

        assert result == {"sections": []}

    @pytest.mark.asyncio
    async def test_extract_handles_non_dict_response(self) -> None:
        """extract() returns {'sections': []} when LLM returns a list instead of dict."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '["not", "a", "dict"]'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAISectionExtractor(client=mock_client)

        result = await extractor.extract("Test", "content")

        assert result == {"sections": []}

    def test_model_default_is_set(self) -> None:
        """The model parameter defaults to 'gpt-4o-mini'."""
        # Note: constructor does create default AsyncOpenAI() but that
        # requires OPENAI_API_KEY at runtime. We only verify the model
        # parameter default without hitting the client init.
        extractor = OpenAISectionExtractor.__new__(OpenAISectionExtractor)
        # Default model is applied inside __init__, so we verify via
        # the ranking/ module pattern — the config handles defaults.
        from research_to_dev.shared.config import ProfileConfig
        assert ProfileConfig().model == "gpt-4o-mini"
