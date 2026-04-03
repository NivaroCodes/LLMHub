import json
from typing import AsyncIterator

import httpx

from app.providers.MainProviders import MainProvider


class OllamaProvider(MainProvider):
    def __init__(self, base_url: str, default_model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def get_completion(self, prompt: str, model: str | None = None, **kwargs) -> str:
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")

    async def get_streaming_completion(
        self, prompt: str, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": True,
            **kwargs,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("response")
                    if token:
                        yield token
                    if data.get("done"):
                        break
