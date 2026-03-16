"""Multi-model adapter for querying different LLM providers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelResponse:
    """Structured response from a model query."""

    text: str
    model_name: str
    usage: dict[str, int] | None = None


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def query(self, prompt: str, system: Optional[str] = None) -> ModelResponse:
        """Send a prompt to the model and return its response."""


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        import anthropic

        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"

    def query(self, prompt: str, system: Optional[str] = None) -> ModelResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        return ModelResponse(
            text=text,
            model_name=self.name,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        import openai

        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    def query(self, prompt: str, system: Optional[str] = None) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return ModelResponse(text=text, model_name=self.name, usage=usage)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    "claude": AnthropicAdapter,
    "anthropic": AnthropicAdapter,
    "gpt-4": OpenAIAdapter,
    "gpt-4o": OpenAIAdapter,
    "openai": OpenAIAdapter,
}


def get_model(name: str, **kwargs) -> ModelAdapter:
    """Resolve a short model name to an adapter instance.

    Parameters
    ----------
    name:
        A key from the adapter registry (e.g. ``"claude"``, ``"gpt-4o"``).
    **kwargs:
        Forwarded to the adapter constructor.
    """
    key = name.lower().strip()
    adapter_cls = _ADAPTER_REGISTRY.get(key)
    if adapter_cls is None:
        available = ", ".join(sorted(_ADAPTER_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Available adapters: {available}"
        )
    return adapter_cls(**kwargs)
