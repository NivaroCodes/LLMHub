import asyncio
import os
from typing import AsyncIterator

from google import genai

from app.providers.MainProviders import MainProvider


class GeminiProvider(MainProvider):
    def __init__(self, api_key: str, default_model: str = "gemini-3-flash-preview"):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        self.client = genai.Client()
        self.default_model = default_model

    async def get_completion(self, prompt: str, model: str | None = None, **kwargs) -> str:
        def _call() -> str:
            response = self.client.models.generate_content(
                model=model or self.default_model,
                contents=prompt,
            )
            return getattr(response, "text", "") or ""

        return await asyncio.to_thread(_call)

    async def get_streaming_completion(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        text = await self.get_completion(prompt, model=model, **kwargs)
        if text:
            yield text
