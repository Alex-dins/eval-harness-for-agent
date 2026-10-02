"""Provider-agnostic model client.

The agent talks to an LLM only through the ``ModelClient`` protocol, so the loop
in ``agent.py`` never imports a vendor SDK directly. The default implementation is
``OpenAIModel``; add another class implementing ``complete`` to swap providers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """Normalized model output, independent of any provider's schema."""

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class ModelClient(Protocol):
    """Anything the agent can use as its brain."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        ...


class OpenAIModel:
    """OpenAI chat-completions implementation of ``ModelClient``.

    Imports the OpenAI client from ``langfuse.openai`` so that every request is
    automatically traced when Langfuse credentials are present, and behaves like a
    plain OpenAI client when they are not.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        # Use the Langfuse-traced OpenAI client only when Langfuse is configured;
        # otherwise the plain client, so we never attempt to export traces.
        from .observability import langfuse_enabled

        if langfuse_enabled():
            from langfuse.openai import OpenAI
        else:
            from openai import OpenAI

        self.model = model or os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self._client = OpenAI()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": call.function.arguments}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        return ModelResponse(
            text=message.content,
            tool_calls=tool_calls,
            raw=response,
        )
