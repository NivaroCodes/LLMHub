import asyncio
import logging
import os
from typing import AsyncIterator

from google import genai

from app.core.canonical import CanonicalChatRequest
from app.providers.base import BaseChatProvider

logger = logging.getLogger(__name__)


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
        timeout_s = kwargs.pop("timeout_s", 7.0) # Default to 7s if not provided
        prompt = self._request_to_prompt(request)

        # Gemini SDK generate_content supports a timeout via request_options or similar, 
        # but the simplest way is to ensure we don't hang the thread pool.
        def _call() -> str:
            response = self.client.models.generate_content(
                model=model or self.default_model,
                contents=prompt,
                # Some SDK versions support config={'timeout': timeout_s}
            )
            return getattr(response, "text", "") or ""

        try:
            # We wrap the thread call in another wait_for just in case the thread hangs
            return await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout_s + 0.5)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("[GEMINI] Timeout after %ss", timeout_s)
            raise TimeoutError(f"Gemini timeout {timeout_s}s")
        except asyncio.CancelledError:
            logger.info("[GEMINI] Request cancelled")
            raise

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
