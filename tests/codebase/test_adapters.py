"""Unit tests for OpenAIDescriber adapter.

Uses ``AsyncMock`` to verify API call shapes without real API calls.
No ``OPENAI_API_KEY`` required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.codebase.adapters import (
    OpenAIDescriber,
    _degraded_result,
    _is_trivial_docstring,
)


# ==================================================================
# _is_trivial_docstring helper
# ==================================================================


class TestIsTrivialDocstring:
    """Verify docstring triviality checks (CA-06)."""

    def test_short_string_is_trivial(self) -> None:
        assert _is_trivial_docstring("Test.") is True
        assert _is_trivial_docstring("Hi") is True

    def test_constructor_is_trivial(self) -> None:
        assert _is_trivial_docstring("Constructor.") is True
        assert _is_trivial_docstring("Initialize self.") is True

    def test_useful_docstring_is_not_trivial(self) -> None:
        assert _is_trivial_docstring("Trains the model on the given dataset.") is False
        assert _is_trivial_docstring("A comprehensive survey of techniques.") is False

    def test_exactly_ten_chars_is_trivial(self) -> None:
        assert _is_trivial_docstring("123456789") is True  # 9 chars
        assert _is_trivial_docstring("1234567890") is False  # 10 chars


# ==================================================================
# _degraded_result helper
# ==================================================================


class TestDegradedResult:
    """Verify degraded output shape (CA-07)."""

    def test_returns_empty_descriptions(self) -> None:
        components = [
            {"name": "foo", "kind": "function", "docstring": None},
            {"name": "bar", "kind": "class", "docstring": "Does stuff."},
        ]
        result = _degraded_result(components)
        assert result["module_summary"] == ""
        assert result["key_responsibilities"] == []
        assert len(result["descriptions"]) == 2
        assert result["descriptions"][0]["description"] == ""
        # Falls back to existing docstring when available
        assert result["descriptions"][1]["description"] == "Does stuff."

    def test_empty_components(self) -> None:
        result = _degraded_result([])
        assert result["descriptions"] == []
        assert result["module_summary"] == ""
        assert result["key_responsibilities"] == []


# ==================================================================
# OpenAIDescriber
# ==================================================================


class TestOpenAIDescriber:
    """Verify OpenAIDescriber call shape and error handling."""

    @pytest.mark.asyncio
    async def test_describe_sends_correct_prompt(self) -> None:
        """describe() sends full source + components with json_object response_format."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = (
            '{"descriptions": ['
            '{"name": "foo", "kind": "function", "description": "Does foo."}'
            '], "module_summary": "A utility module.", '
            '"key_responsibilities": ["Utility functions"]}'
        )
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        describer = OpenAIDescriber(client=mock_client, model="gpt-4o-mini")

        result = await describer.describe(
            "utils.helpers",
            "def foo():\n    pass\n",
            [{"name": "foo", "kind": "function", "signature": "def foo()", "docstring": None}],
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
        assert "utils.helpers" in messages[1]["content"]
        assert "def foo()" in messages[1]["content"]

        # Verify parsing
        assert len(result["descriptions"]) == 1
        assert result["descriptions"][0]["description"] == "Does foo."
        assert result["module_summary"] == "A utility module."
        assert result["key_responsibilities"] == ["Utility functions"]

    @pytest.mark.asyncio
    async def test_describe_preserves_good_docstrings(self) -> None:
        """CA-06: Good docstrings are included in the prompt as existing_docstring."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"descriptions": [], "module_summary": "", "key_responsibilities": []}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        describer = OpenAIDescriber(client=mock_client)

        await describer.describe(
            "mod",
            "source",
            [{
                "name": "train",
                "kind": "function",
                "signature": "def train():",
                "docstring": "Trains the model using gradient descent.",
            }],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        # The good docstring should be in the prompt
        assert "existing_docstring" in user_content
        assert "Trains the model using gradient descent." in user_content

    @pytest.mark.asyncio
    async def test_describe_skips_trivial_docstrings(self) -> None:
        """CA-06: Trivial docstrings are NOT passed as existing_docstring."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"descriptions": [], "module_summary": "", "key_responsibilities": []}'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        describer = OpenAIDescriber(client=mock_client)

        await describer.describe(
            "mod",
            "source",
            [{
                "name": "foo",
                "kind": "function",
                "signature": "def foo():",
                "docstring": "Constructor.",
            }],
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        # Trivial docstring should NOT appear as existing_docstring
        assert "existing_docstring" not in user_content

    @pytest.mark.asyncio
    async def test_describe_handles_none_content(self) -> None:
        """CA-07: Empty LLM content → degraded result."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        describer = OpenAIDescriber(client=mock_client)

        result = await describer.describe(
            "mod",
            "source",
            [{"name": "foo", "kind": "function", "docstring": None}],
        )

        assert result["module_summary"] == ""
        assert result["key_responsibilities"] == []
        assert len(result["descriptions"]) == 1
        assert result["descriptions"][0]["description"] == ""

    @pytest.mark.asyncio
    async def test_describe_handles_json_parse_error(self) -> None:
        """CA-07: Invalid JSON → degraded result."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = "not valid json!!!"
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        describer = OpenAIDescriber(client=mock_client)

        result = await describer.describe(
            "mod",
            "source",
            [{"name": "f", "kind": "function", "docstring": None}],
        )

        assert result["descriptions"][0]["description"] == ""
        assert result["module_summary"] == ""

    @pytest.mark.asyncio
    async def test_describe_handles_api_exception(self) -> None:
        """CA-07: LLM call raises → degraded result."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "API connection error"
        )

        describer = OpenAIDescriber(client=mock_client)

        result = await describer.describe(
            "mod",
            "source",
            [{"name": "f", "kind": "function", "docstring": None}],
        )

        assert result["descriptions"][0]["description"] == ""
        assert result["module_summary"] == ""

    @pytest.mark.asyncio
    async def test_describe_handles_non_dict_response(self) -> None:
        """CA-07: LLM returns a list instead of dict → degraded result."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '["not", "a", "dict"]'
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        describer = OpenAIDescriber(client=mock_client)

        result = await describer.describe(
            "mod",
            "source",
            [{"name": "f", "kind": "function", "docstring": None}],
        )

        assert result["descriptions"][0]["description"] == ""

    @pytest.mark.asyncio
    async def test_describe_fills_missing_keys(self) -> None:
        """Missing keys in valid JSON are filled with defaults."""
        mock_client = AsyncMock()

        mock_choice = AsyncMock()
        mock_message = AsyncMock()
        mock_message.content = '{"other": "stuff"}'  # no descriptions, module_summary, etc.
        mock_choice.message = mock_message
        mock_response = AsyncMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        describer = OpenAIDescriber(client=mock_client)

        result = await describer.describe(
            "mod",
            "source",
            [{"name": "f", "kind": "function", "docstring": None}],
        )

        assert result["descriptions"] == []
        assert result["module_summary"] == ""
        assert result["key_responsibilities"] == []

    @pytest.mark.asyncio
    async def test_model_default_is_set(self) -> None:
        """The model parameter defaults to 'gpt-4o-mini'."""
        from research_to_dev.shared.config import CodebaseConfig

        assert CodebaseConfig().model == "gpt-4o-mini"
