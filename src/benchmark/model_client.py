"""Abstraction for model generation and judging with configurable providers."""

from __future__ import annotations

import os
from typing import Protocol

from benchmark.config import ModelEntry
from benchmark.exceptions import FatalBenchmarkError
from benchmark.types import GenerateResult, JudgeResult


class ModelClient(Protocol):
    """Protocol for model clients that can generate and judge."""

    async def generate(
        self, model_name: str, system_prompt: str, user_message: str
    ) -> GenerateResult: ...

    async def judge(
        self, model_name: str, system_prompt: str, user_message: str
    ) -> GenerateResult: ...


class OpenRouterClient:
    """OpenRouter-backed model client."""

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise FatalBenchmarkError(
                "OPENROUTER_API_KEY environment variable not set."
            )

    async def generate(
        self, model_name: str, system_prompt: str, user_message: str
    ) -> GenerateResult:
        from openai import AsyncOpenAI
        from benchmark.utils import openai_compat_generate

        model = ModelEntry(name=model_name)
        async with AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
        ) as client:
            return await openai_compat_generate(
                client, model, system_prompt, user_message
            )

    async def judge(
        self, model_name: str, system_prompt: str, user_message: str
    ) -> GenerateResult:
        return await self.generate(model_name, system_prompt, user_message)


class GeminiClient:
    """Google Gemini-backed model client using google-genai SDK."""

    def __init__(self) -> None:
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise FatalBenchmarkError(
                "GEMINI_API_KEY environment variable not set."
            )

    async def generate(
        self, model_name: str, system_prompt: str, user_message: str
    ) -> GenerateResult:
        from google import genai

        client = genai.Client(api_key=self._api_key)
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=user_message,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        return GenerateResult(
            response=response.text or "",
            raw_api_response={"model": model_name, "text": response.text},
        )

    async def judge(
        self, model_name: str, system_prompt: str, user_message: str
    ) -> GenerateResult:
        return await self.generate(model_name, system_prompt, user_message)


def get_model_client(provider: str) -> OpenRouterClient | GeminiClient:
    """Factory for model clients based on provider name."""
    if provider == "openrouter":
        return OpenRouterClient()
    elif provider == "gemini":
        return GeminiClient()
    else:
        raise FatalBenchmarkError(
            f"Unknown provider: {provider!r}. Supported: 'openrouter', 'gemini'"
        )
