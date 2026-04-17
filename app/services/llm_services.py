import asyncio
import os
import time
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import HTTPException

from app.db.database import log_request
from app.providers.gemini_client import GeminiProvider
from app.providers.openai_client import OpenAIProvider
from app.providers.openrouter_client import OpenRouterProvider
from app.providers.ollama_client import OllamaProvider
from app.schemas.chat import ChatRequest
from app.services.cache_service import get_cached_response, set_cached_response
from app.services.cost_service import estimate_cost

_MAX_RETRIES = 2
_RETRY_DELAY_S = 0.5


class LLMService:
    def __init__(self):
        self.openai_api_key     = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key     = os.getenv("GEMINI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.ollama_base_url    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.openai_model     = os.getenv("OPENAI_MODEL",     "gpt-4o-mini")
        self.gemini_model     = os.getenv("GEMINI_MODEL",     "gemini-2.0-flash")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.ollama_model     = os.getenv("OLLAMA_MODEL",     "llama3:8b")

        self.router_mode  = os.getenv("ROUTER_MODE",  "rules").strip().lower()
        self.router_model = os.getenv("ROUTER_MODEL", self.gemini_model)

        self.openai     = OpenAIProvider(api_key=self.openai_api_key) if self.openai_api_key else None
        self.gemini     = GeminiProvider(api_key=self.gemini_api_key, default_model=self.gemini_model) if self.gemini_api_key else None
        self.openrouter = OpenRouterProvider(api_key=self.openrouter_api_key) if self.openrouter_api_key else None
        self.ollama     = OllamaProvider(base_url=self.ollama_base_url, default_model=self.ollama_model)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _model_for(self, provider_name: str) -> str:
        return {
            "gemini":     self.gemini_model,
            "openai":     self.openai_model,
            "openrouter": self.openrouter_model,
            "ollama":     self.ollama_model,
        }.get(provider_name, "unknown")

    def _rule_route(self, user_text: str) -> str:
        text = user_text.lower()
        if len(user_text) > 160:
            return "gemini"
        if any(tok in text for tok in ("код", "code", "алгоритм", "объясни", "сравни", "план")):
            return "gemini"
        return "ollama"

    async def _agent_route(self, user_text: str) -> str:
        if self.gemini is None:
            return self._rule_route(user_text)
        prompt = (
            "You are a routing agent for an LLM gateway. "
            "Choose the best provider for the user request. "
            "Return ONLY one word: gemini or ollama.\n\n"
            f"User request:\n{user_text}"
        )
        try:
            decision = await self.gemini.get_completion(prompt, model=self.router_model)
        except Exception:
            return self._rule_route(user_text)
        value = decision.strip().lower()
        if "ollama" in value:
            return "ollama"
        if "gemini" in value:
            return "gemini"
        return self._rule_route(user_text)

    def _get_provider_chain(self, preferred: str) -> list[tuple[str, object]]:
        """Return ordered list of (name, provider) to try, skipping unavailable providers."""
        def _add(chain, name, instance):
            if instance is not None:
                chain.append((name, instance))

        chain: list[tuple[str, object]] = []
        if preferred == "gemini":
            _add(chain, "gemini",     self.gemini)
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "ollama",     self.ollama)
        elif preferred == "openai":
            _add(chain, "openai",     self.openai)
            _add(chain, "gemini",     self.gemini)
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "ollama",     self.ollama)
        elif preferred == "openrouter":
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "gemini",     self.gemini)
            _add(chain, "ollama",     self.ollama)
        elif preferred == "ollama":
            _add(chain, "ollama",     self.ollama)
            _add(chain, "gemini",     self.gemini)
        else:  # auto
            _add(chain, "gemini",     self.gemini)
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "ollama",     self.ollama)
        return chain

    # ------------------------------------------------------------------
    # Provider call (unified interface)
    # ------------------------------------------------------------------

    async def _call_provider(
        self,
        provider_name: str,
        provider: object,
        user_text: str,
        timeout_s: int,
    ) -> str:
        model = self._model_for(provider_name)
        if provider_name == "ollama":
            return await provider.get_completion(user_text, model=model, timeout_s=timeout_s)
        return await provider.get_completion(user_text, model=model)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def get_response(self, payload: "ChatRequest | str") -> dict:
        start      = time.monotonic()
        request_id = str(uuid.uuid4())
        timestamp  = datetime.now(timezone.utc)

        if isinstance(payload, ChatRequest):
            user_text = payload.message
            preferred = (payload.preferred_provider or "auto").strip().lower()
            timeout_s = max(1, int(payload.timeout_ms / 1000))
        else:
            user_text = str(payload)
            preferred = "auto"
            timeout_s = 30

        if preferred == "auto":
            preferred = (
                await self._agent_route(user_text)
                if self.router_mode == "agent"
                else self._rule_route(user_text)
            )

        chain = self._get_provider_chain(preferred)
        if not chain:
            raise HTTPException(status_code=500, detail="No providers available")

        # ---- Cache check ----
        cache_provider = chain[0][0]
        cache_model    = self._model_for(cache_provider)
        try:
            cached = await get_cached_response(user_text, cache_provider, cache_model)
        except Exception:
            cached = None

        if cached is not None:
            latency_ms = int((time.monotonic() - start) * 1000)
            prov = cached["provider"]
            mdl  = cached["model"]
            prompt_tokens, completion_tokens, _ = estimate_cost(prov, mdl, user_text, cached["answer"])
            try:
                await log_request(
                    request_id=request_id, timestamp=timestamp, message=user_text,
                    provider=prov, model=mdl, latency_ms=latency_ms,
                    cached=True, fallback_used=cached["fallback_used"],
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    cost_usd=0.0, status="ok",
                )
            except Exception:
                pass
            return {
                "answer": cached["answer"], "provider": prov, "model": mdl,
                "latency_ms": latency_ms, "request_id": request_id,
                "fallback_used": cached["fallback_used"], "cached": True, "cost_usd": 0.0,
            }

        # ---- Provider calls with retry + fallback ----
        answer        = None
        provider_name = None
        fallback_used = False
        last_error    = None

        for i, (prov_name, prov) in enumerate(chain):
            for attempt in range(_MAX_RETRIES):
                try:
                    answer        = await self._call_provider(prov_name, prov, user_text, timeout_s)
                    provider_name = prov_name
                    fallback_used = i > 0
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404 and prov_name == "ollama":
                        model_name = self._model_for("ollama")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Ollama model '{model_name}' not found. Run: ollama pull {model_name}",
                        ) from exc
                    last_error = exc
                except Exception as exc:
                    last_error = exc

                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_DELAY_S * (attempt + 1))

            if answer is not None:
                break

        latency_ms = int((time.monotonic() - start) * 1000)
        model_used = self._model_for(provider_name) if provider_name else "unknown"

        # ---- All providers failed ----
        if answer is None:
            error_msg = str(last_error) if last_error else "All providers failed"
            try:
                await log_request(
                    request_id=request_id, timestamp=timestamp, message=user_text,
                    provider=chain[0][0], model=self._model_for(chain[0][0]),
                    latency_ms=latency_ms, cached=False, fallback_used=False,
                    prompt_tokens=None, completion_tokens=None, cost_usd=None,
                    status="error", error=error_msg,
                )
            except Exception:
                pass
            raise HTTPException(status_code=502, detail=f"All providers failed: {error_msg}")

        # ---- Cost + cache store + log ----
        prompt_tokens, completion_tokens, cost_usd = estimate_cost(
            provider_name, model_used, user_text, answer
        )

        try:
            await set_cached_response(
                user_text, provider_name, model_used,
                {"answer": answer, "provider": provider_name, "model": model_used, "fallback_used": fallback_used},
            )
        except Exception:
            pass

        try:
            await log_request(
                request_id=request_id, timestamp=timestamp, message=user_text,
                provider=provider_name, model=model_used, latency_ms=latency_ms,
                cached=False, fallback_used=fallback_used,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                cost_usd=cost_usd, status="ok",
            )
        except Exception:
            pass

        return {
            "answer": answer, "provider": provider_name, "model": model_used,
            "latency_ms": latency_ms, "request_id": request_id,
            "fallback_used": fallback_used, "cached": False, "cost_usd": cost_usd,
        }
