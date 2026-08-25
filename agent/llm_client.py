"""
agent/llm_client.py
====================
Provider-agnostic LLM client for Groq and Gemini.

Usage:
    client = get_llm_client()  # reads LLM_PROVIDER from env
    response = client.chat(messages=[...], tools=[...], system="...")
    print(response.text, response.tool_calls, response.stop_reason)
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """Normalized tool-call shape, provider-agnostic."""
    name: str
    arguments: dict[str, Any]
    id: str = ""          # Groq/OpenAI supply an id; Gemini does not


@dataclass
class LLMResponse:
    """Normalized response from any provider."""
    text: str                              # assistant text (may be "" if tool_calls present)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"             # "stop" | "tool_calls" | "length" | "error"
    raw: Any = None                        # original provider response (for debugging)
    _raw_content: Any = None               # provider-native Content for history (Gemini only)


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────────────────────

def _with_retry(fn, max_retries: int = 4, base_delay: float = 2.0):
    """
    Call *fn()* with exponential backoff on rate-limit (429) errors.
    Logs each retry so eval sweeps can surface rate-limit pressure.
    """
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = (
                "429" in err_str
                or "rate_limit" in err_str.lower()
                or "quota" in err_str.lower()
                or "resource_exhausted" in err_str.lower()
            )
            if is_rate_limit and attempt < max_retries:
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %.1fs. Error: %s",
                    attempt + 1, max_retries, delay, err_str,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)   # cap at 60 s
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """
    Abstract base class for LLM providers.

    Subclasses must implement `chat()`.
    `tools` is a list of OpenAI-style JSON-schema dicts:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Groq implementation
# ─────────────────────────────────────────────────────────────────────────────

class GroqClient(LLMClient):
    """
    Groq provider using the official `groq` SDK.
    Normalises the OpenAI-compatible tool_calls response into ToolCall objects.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq  # type: ignore
        except ImportError as e:
            raise ImportError("Install the 'groq' package: pip install groq") from e

        self._client = Groq(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        # Prepend system message if provided and not already present
        full_messages = list(messages)
        if system and (not full_messages or full_messages[0].get("role") != "system"):
            full_messages = [{"role": "system", "content": system}] + full_messages

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        def _call():
            return self._client.chat.completions.create(**kwargs)

        raw = _with_retry(_call)
        choice = raw.choices[0]
        msg = choice.message
        stop_reason = choice.finish_reason or "stop"

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            stop_reason = "tool_calls"
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=args,
                    id=tc.id or "",
                ))

        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=raw,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gemini implementation
# ─────────────────────────────────────────────────────────────────────────────

class GeminiClient(LLMClient):
    """
    Gemini provider using the `google-genai` SDK.
    Uses manual function-calling (AFC disabled) so the agent loop stays in control.
    Converts OpenAI-style JSON-schema tool dicts → google.genai FunctionDeclarations.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install the 'google-genai' package: pip install google-genai"
            ) from e

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model

    @staticmethod
    def _oai_schema_to_gemini(tools: list[dict]) -> list:
        """Convert OpenAI-style tool list to google.genai FunctionDeclaration objects."""
        from google.genai import types  # type: ignore

        declarations = []
        for tool in tools:
            fn = tool.get("function", {})
            params = fn.get("parameters", {})
            # google-genai accepts Schema dict directly
            declarations.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=params,
                )
            )
        return declarations

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        from google.genai import types  # type: ignore

        # --- Build Gemini Contents from OpenAI-style message history ---
        # We keep a list of (Content | raw_content) where any assistant turn
        # that came from a previous Gemini response has its raw Content already
        # stored in msg["_gemini_raw_content"] so we can pass it verbatim and
        # preserve thought_signature from thinking models.
        contents = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                i += 1
                continue

            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content or "")]
                ))
                i += 1

            elif role == "assistant":
                # If we stored the raw Gemini Content from a previous turn, use it
                # verbatim so thought_signature is preserved.
                if "_gemini_raw_content" in msg:
                    contents.append(msg["_gemini_raw_content"])
                else:
                    # Reconstruct from dict (first turn or non-Gemini history)
                    parts = []
                    if content:
                        parts.append(types.Part.from_text(text=content))
                    if msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            fn_name = tc.get("function", {}).get("name", "")
                            try:
                                fn_args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                            except json.JSONDecodeError:
                                fn_args = {}
                            parts.append(types.Part.from_function_call(
                                name=fn_name, args=fn_args
                            ))
                    if not parts:
                        parts = [types.Part.from_text(text="")]
                    contents.append(types.Content(role="model", parts=parts))
                i += 1

            elif role == "tool":
                # Collect all consecutive tool results into one user Content
                tool_parts = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tmsg = messages[i]
                    tool_name = tmsg.get("name", tmsg.get("tool_call_id", ""))
                    result_content = tmsg.get("content", "")
                    if isinstance(result_content, str):
                        result_content = {"result": result_content}
                    tool_parts.append(types.Part.from_function_response(
                        name=tool_name,
                        response=result_content,
                    ))
                    i += 1
                contents.append(types.Content(role="user", parts=tool_parts))
            else:
                i += 1

        # Build config
        config_kwargs: dict[str, Any] = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            gemini_tools = [types.Tool(
                function_declarations=self._oai_schema_to_gemini(tools)
            )]
            config_kwargs["tools"] = gemini_tools

        config = types.GenerateContentConfig(**config_kwargs)

        def _call():
            return self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        raw = _with_retry(_call)

        # Parse response — look for function_call parts first.
        # Thinking models (gemini-3.6-flash) can emit "thought" parts (internal
        # reasoning) alongside or instead of a text/function_call part.  We must
        # skip thought-only parts when building the visible text, but fall back
        # to them if *nothing else* was produced so we never return empty.
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        thought_parts: list[str] = []

        if raw.candidates:
            candidate = raw.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # function_call wins regardless
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_calls.append(ToolCall(
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                            id="",
                        ))
                    elif hasattr(part, "text") and part.text:
                        # Gemini thinking models mark internal thoughts via
                        # part.thought == True; keep them separate so we can
                        # fall back to them if no visible text exists.
                        if getattr(part, "thought", False):
                            thought_parts.append(part.text)
                        else:
                            text_parts.append(part.text)

        # If the model only produced thought with no visible text/tool call,
        # surface the thought text rather than returning empty.
        if not text_parts and not tool_calls and thought_parts:
            logger.debug("Gemini returned thought-only; surfacing thought text.")
            text_parts = thought_parts

        stop_reason = "tool_calls" if tool_calls else "stop"
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=raw,
            _raw_content=raw.candidates[0].content if raw.candidates else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_client(provider: str | None = None) -> LLMClient:
    """
    Factory that reads LLM_PROVIDER (env) and returns the appropriate client.
    Fails immediately with a clear message if the required API key is missing.

    Args:
        provider: Override the env var. Values: "groq" | "gemini".

    Returns:
        Configured LLMClient subclass.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return GroqClient(api_key=api_key, model=model)

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        return GeminiClient(api_key=api_key, model=model)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Must be 'groq' or 'gemini'."
        )
