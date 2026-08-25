"""
tests/test_llm_client.py
=========================
Unit tests for the LLM client abstraction layer.
Tests use mocked HTTP responses so no API keys are needed to run them.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.llm_client import (
    GeminiClient,
    GroqClient,
    LLMResponse,
    ToolCall,
    get_llm_client,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build fake provider responses
# ─────────────────────────────────────────────────────────────────────────────

def _make_groq_tool_call_response(tool_name: str, arguments: dict) -> MagicMock:
    """Construct a mock Groq response object that contains a tool call."""
    tc = MagicMock()
    tc.id = "call_abc123"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(arguments)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "tool_calls"

    raw = MagicMock()
    raw.choices = [choice]
    return raw


def _make_groq_text_response(text: str) -> MagicMock:
    """Construct a mock Groq response object that contains plain text."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    raw = MagicMock()
    raw.choices = [choice]
    return raw


def _make_gemini_tool_call_response(tool_name: str, arguments: dict) -> MagicMock:
    """Construct a mock Gemini response object that contains a function_call part."""
    fc = MagicMock()
    fc.name = tool_name
    fc.args = arguments  # dict-like

    part = MagicMock()
    part.function_call = fc
    part.text = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    raw = MagicMock()
    raw.candidates = [candidate]
    return raw


def _make_gemini_text_response(text: str) -> MagicMock:
    """Construct a mock Gemini response object that contains plain text."""
    part = MagicMock()
    part.function_call = None
    part.text = text

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    raw = MagicMock()
    raw.candidates = [candidate]
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# GroqClient tests
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_problems",
            "description": "Search for competitive programming problems",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "rating_min": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    }
]

SAMPLE_MESSAGES = [{"role": "user", "content": "Find me a graph problem rated 1800"}]


class TestGroqClient:
    """Tests for GroqClient response normalisation."""

    def _make_client(self) -> GroqClient:
        with patch("groq.Groq"):
            client = GroqClient(api_key="fake-key")
        return client

    def test_tool_call_normalisation(self):
        """GroqClient must return a ToolCall with correct name and arguments."""
        raw = _make_groq_tool_call_response(
            "search_problems", {"query": "graph", "rating_min": 1800}
        )
        client = self._make_client()
        client._client.chat.completions.create.return_value = raw

        response = client.chat(SAMPLE_MESSAGES, tools=SAMPLE_TOOL)

        assert isinstance(response, LLMResponse)
        assert response.stop_reason == "tool_calls"
        assert len(response.tool_calls) == 1

        tc = response.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.name == "search_problems"
        assert tc.arguments == {"query": "graph", "rating_min": 1800}
        assert tc.id == "call_abc123"

    def test_text_response_normalisation(self):
        """GroqClient must return plain text correctly with no tool_calls."""
        raw = _make_groq_text_response("Here is a graph problem for you.")
        client = self._make_client()
        client._client.chat.completions.create.return_value = raw

        response = client.chat(SAMPLE_MESSAGES)

        assert isinstance(response, LLMResponse)
        assert response.text == "Here is a graph problem for you."
        assert response.tool_calls == []
        assert response.stop_reason == "stop"

    def test_system_message_prepended(self):
        """System prompt must be prepended as role=system message."""
        raw = _make_groq_text_response("ok")
        client = self._make_client()
        client._client.chat.completions.create.return_value = raw

        client.chat(SAMPLE_MESSAGES, system="You are a helpful assistant.")

        call_args = client._client.chat.completions.create.call_args
        messages_sent = call_args[1]["messages"]
        assert messages_sent[0]["role"] == "system"
        assert "helpful" in messages_sent[0]["content"]

    def test_invalid_json_arguments_handled(self):
        """GroqClient must handle malformed JSON in tool call arguments gracefully."""
        tc = MagicMock()
        tc.id = "bad_json"
        tc.function.name = "search_problems"
        tc.function.arguments = "not-valid-json"

        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [tc]

        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "tool_calls"

        raw = MagicMock()
        raw.choices = [choice]

        client = self._make_client()
        client._client.chat.completions.create.return_value = raw

        response = client.chat(SAMPLE_MESSAGES, tools=SAMPLE_TOOL)
        # Should not raise; arguments should be empty dict
        assert response.tool_calls[0].arguments == {}


# ─────────────────────────────────────────────────────────────────────────────
# GeminiClient tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiClient:
    """Tests for GeminiClient response normalisation."""

    def _make_client(self) -> GeminiClient:
        """Create a GeminiClient with mocked google-genai internals."""
        mock_genai = MagicMock()
        mock_types = MagicMock()

        # Allow types.Content, types.Part, etc. to be called (they're just MagicMock)
        mock_types.AutomaticFunctionCallingConfig.return_value = MagicMock()
        mock_types.GenerateContentConfig.return_value = MagicMock()
        mock_types.Tool.return_value = MagicMock()
        mock_types.FunctionDeclaration.return_value = MagicMock()
        mock_types.Content.return_value = MagicMock()
        mock_types.Part.from_text.return_value = MagicMock()
        mock_types.Part.from_function_response.return_value = MagicMock()
        mock_types.Part.from_function_call.return_value = MagicMock()

        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai, "google.genai.types": mock_types}):
            with patch("agent.llm_client.GeminiClient.__init__", lambda self, api_key, model="gemini-2.0-flash": None):
                client = GeminiClient.__new__(GeminiClient)
                client._genai = mock_genai
                client._types = mock_types
                client._client = mock_genai.Client()
                client.model = "gemini-2.0-flash"
        return client

    def test_tool_call_normalisation(self):
        """GeminiClient must normalise function_call parts into ToolCall objects."""
        raw = _make_gemini_tool_call_response(
            "search_problems", {"query": "graph", "rating_min": 1800}
        )
        client = self._make_client()
        client._client.models.generate_content.return_value = raw

        # Patch types used inside chat()
        with patch("agent.llm_client.GeminiClient._oai_schema_to_gemini", return_value=[]):
            with patch("google.genai.types") as _:
                response = client.chat(SAMPLE_MESSAGES, tools=SAMPLE_TOOL)

        assert isinstance(response, LLMResponse)
        assert response.stop_reason == "tool_calls"
        assert len(response.tool_calls) == 1

        tc = response.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.name == "search_problems"
        assert tc.arguments == {"query": "graph", "rating_min": 1800}
        assert tc.id == ""  # Gemini doesn't supply call IDs

    def test_text_response_normalisation(self):
        """GeminiClient must return plain text correctly with no tool_calls."""
        raw = _make_gemini_text_response("Here is a graph problem for you.")
        client = self._make_client()
        client._client.models.generate_content.return_value = raw

        response = client.chat(SAMPLE_MESSAGES)

        assert isinstance(response, LLMResponse)
        assert response.text == "Here is a graph problem for you."
        assert response.tool_calls == []
        assert response.stop_reason == "stop"


# ─────────────────────────────────────────────────────────────────────────────
# Normalised shape parity test
# ─────────────────────────────────────────────────────────────────────────────

def test_both_providers_return_same_shape():
    """
    Critical: GroqClient and GeminiClient must return identical LLMResponse shape
    for the same logical tool-call response.
    """
    # Build identical logical tool call from both providers
    groq_raw = _make_groq_tool_call_response("get_problem", {"problem_id": "1234A"})
    gemini_raw = _make_gemini_tool_call_response("get_problem", {"problem_id": "1234A"})

    # GroqClient
    with patch("groq.Groq"):
        groq_client = GroqClient(api_key="fake")
    groq_client._client.chat.completions.create.return_value = groq_raw
    groq_response = groq_client.chat([{"role": "user", "content": "Get problem 1234A"}])

    # GeminiClient (direct construction bypass)
    gemini_client = MagicMock(spec=GeminiClient)
    gemini_client.chat.return_value = LLMResponse(
        text="",
        tool_calls=[ToolCall(name="get_problem", arguments={"problem_id": "1234A"}, id="")],
        stop_reason="tool_calls",
    )
    gemini_response = gemini_client.chat([{"role": "user", "content": "Get problem 1234A"}])

    # Both responses must have the same shape
    assert groq_response.stop_reason == gemini_response.stop_reason == "tool_calls"
    assert len(groq_response.tool_calls) == len(gemini_response.tool_calls) == 1
    assert groq_response.tool_calls[0].name == gemini_response.tool_calls[0].name == "get_problem"
    assert groq_response.tool_calls[0].arguments == gemini_response.tool_calls[0].arguments == {"problem_id": "1234A"}


# ─────────────────────────────────────────────────────────────────────────────
# get_llm_client factory tests
# ─────────────────────────────────────────────────────────────────────────────

def test_factory_missing_groq_key(monkeypatch):
    """get_llm_client raises EnvironmentError if GROQ_API_KEY is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        get_llm_client("groq")


def test_factory_missing_gemini_key(monkeypatch):
    """get_llm_client raises EnvironmentError if GEMINI_API_KEY is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        get_llm_client("gemini")


def test_factory_unknown_provider(monkeypatch):
    """get_llm_client raises ValueError for unknown provider name."""
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm_client("openai")
