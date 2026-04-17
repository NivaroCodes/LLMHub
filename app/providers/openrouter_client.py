from typing import AsyncIterator

from openai import AsyncOpenAI

from app.providers.MainProviders import MainProvider

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(MainProvider):
    def __init__(self, api_key: str, default_model: str = "openai/gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=_BASE_URL)
        self.default_model = default_model

    async def get_completion(self, prompt: str, model: str | None = None, **kwargs) -> str:
        kwargs.pop("timeout_s", None)
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def get_streaming_completion(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        kwargs.pop("timeout_s", None)
        stream = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
