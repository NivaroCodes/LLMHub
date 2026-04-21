import asyncio
import os
from typing import AsyncIterator

from google import genai

from app.core.canonical import CanonicalChatRequest
from app.providers.base import BaseChatProvider


class GeminiProvider(BaseChatProvider):
    def __init__(self, api_key: str, default_model: str = "gemini-3-flash-preview"):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        self.client = genai.Client()
        self.default_model = default_model

    @staticmethod
    def _request_to_prompt(request: CanonicalChatRequest) -> str:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in request.messages)

    async def chat(self, request: CanonicalChatRequest, model: str | None = None, **kwargs) -> str:
        kwargs.pop("timeout_s", None)
        prompt = self._request_to_prompt(request)

        def _call() -> str:
            response = self.client.models.generate_content(
                model=model or self.default_model,
                contents=prompt,
            )
            return getattr(response, "text", "") or ""

        return await asyncio.to_thread(_call)

    async def stream(
        self, request: CanonicalChatRequest, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        # Fallback streaming for provider without incremental API wiring.
        text = await self.chat(request, model=model, **kwargs)
        if text:
            yield text

    async def get_completion(self, prompt: str, model: str | None = None, **kwargs) -> str:
        request = CanonicalChatRequest.from_text(prompt)
        return await self.chat(request, model=model, **kwargs)

    async def get_streaming_completion(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        request = CanonicalChatRequest.from_text(prompt)
        async for chunk in self.stream(request, model=model, **kwargs):
            yield chunk
