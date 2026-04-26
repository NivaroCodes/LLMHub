import json
from typing import AsyncIterator

import httpx

from app.core.canonical import CanonicalChatRequest
from app.providers.base import BaseChatProvider


class OllamaProvider(BaseChatProvider):
    # For warm local performance, keep model resident after first request.
    # For manual warmup after reboot (PowerShell):
    # Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body '{"model":"ministral-3:3b","prompt":"hi","stream":false,"keep_alive":"24h"}' -ContentType "application/json"
    def __init__(self, base_url: str, default_model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=0.5, read=35.0, write=10.0, pool=1.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    @staticmethod
    def _ollama_payload_from_messages(request: CanonicalChatRequest, model: str) -> dict:
        system_prompt = ""
        prompt_parts: list[str] = []
        for msg in request.messages:
            if msg.role == "system" and not system_prompt:
                system_prompt = msg.content
                continue
            prompt_parts.append(f"{msg.role}: {msg.content}")
        prompt = "\n".join(prompt_parts).strip() or request.user_text
        payload = {"model": model, "prompt": prompt, "keep_alive": "24h"}
        if request.max_tokens is not None:
            payload["options"] = {"num_predict": request.max_tokens}
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    async def chat(self, request: CanonicalChatRequest, model: str | None = None, **kwargs) -> str:
        timeout_s = kwargs.pop("timeout_s", None)
        payload = self._ollama_payload_from_messages(request, model or self.default_model)
        payload["stream"] = False
        payload.update(kwargs)
        timeout = float(timeout_s) if timeout_s is not None else 30.0
        response = await self._client.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "")

    async def stream(
        self, request: CanonicalChatRequest, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        timeout_s = kwargs.pop("timeout_s", None)
        payload = self._ollama_payload_from_messages(request, model or self.default_model)
        payload["stream"] = True
        payload.update(kwargs)
        timeout = float(timeout_s) if timeout_s is not None else 60.0
        async with self._client.stream("POST", f"{self.base_url}/api/generate", json=payload, timeout=timeout) as response:
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
