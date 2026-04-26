from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.core.canonical import CanonicalChatRequest
from app.providers.base import BaseChatProvider

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseChatProvider):
    def __init__(self, api_key: str, default_model: str = "openai/gpt-4o-mini"):
        self.api_key = api_key
        self.default_model = default_model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=2.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, request: CanonicalChatRequest, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": request.to_provider_messages(),
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.user is not None:
            payload["user"] = request.user
        payload.update(kwargs)
        return payload

    async def chat(self, request: CanonicalChatRequest, model: str | None = None, **kwargs) -> str:
        timeout_s = kwargs.pop("timeout_s", None)
        payload = self._payload(request, model=model, **kwargs)
        response = await self._client.post(
            f"{_BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""

    async def stream(
        self, request: CanonicalChatRequest, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        timeout_s = kwargs.pop("timeout_s", None)
        payload = self._payload(request, model=model, stream=True, **kwargs)
        async with self._client.stream(
            "POST",
            f"{_BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=timeout_s,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    yield delta

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_completion(self, prompt: str, model: str | None = None, **kwargs) -> str:
        request = CanonicalChatRequest.from_text(prompt)
        return await self.chat(request, model=model, **kwargs)

    async def get_streaming_completion(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        request = CanonicalChatRequest.from_text(prompt)
        async for chunk in self.stream(request, model=model, **kwargs):
            yield chunk
