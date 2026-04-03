from typing import AsyncIterator

from openai import AsyncOpenAI

from app.providers.MainProviders import MainProvider

class OpenAIProvider(MainProvider):
    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def get_completion(self, prompt: str, model: str | None = None, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def get_streaming_completion(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


class Providers:
    def __init__(self, openai_provider: OpenAIProvider):
        self.openai = openai_provider

    async def generate_response(self, prompt: str):
        return await self.openai.get_completion(prompt)
