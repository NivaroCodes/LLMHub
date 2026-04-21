from typing import AsyncIterator

from openai import AsyncOpenAI

from app.core.canonical import CanonicalChatRequest
from app.providers.base import BaseChatProvider


class OpenAIProvider(BaseChatProvider):
    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def chat(self, request: CanonicalChatRequest, model: str | None = None, **kwargs) -> str:
        kwargs.pop("timeout_s", None)
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=request.to_provider_messages(),
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self, request: CanonicalChatRequest, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=request.to_provider_messages(),
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def get_completion(self, prompt: str, model: str | None = None, **kwargs) -> str:
        request = CanonicalChatRequest.from_text(prompt)
        return await self.chat(request, model=model, **kwargs)

    async def get_streaming_completion(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        request = CanonicalChatRequest.from_text(prompt)
        async for chunk in self.stream(request, model=model, **kwargs):
            yield chunk


class Providers:
    def __init__(self, openai_provider: OpenAIProvider):
        self.openai = openai_provider

    async def generate_response(self, prompt: str):
        return await self.openai.get_completion(prompt)
