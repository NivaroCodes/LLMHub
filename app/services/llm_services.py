import asyncio
import inspect
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from app.core.canonical import CanonicalChatRequest, CanonicalMessage
from app.clients.redis_client import get_redis
from app.core.config import CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_TIMEOUT
from app.core.prompts import get_prompt
from app.db.database import log_request
from app.providers.gemini_client import GeminiProvider
from app.providers.ollama_client import OllamaProvider
from app.providers.openai_client import OpenAIProvider
from app.providers.openrouter_client import OpenRouterProvider
from app.router.rules import choose_route
from app.schemas.chat import ChatRequest
from app.services.cache_service import get_cached_response, set_cached_response
from app.services.cost_service import estimate_cost

_MAX_RETRIES = 2
_RETRY_DELAY_S = 0.5
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3:8b")
        self.router_mode = os.getenv("ROUTER_MODE", "rules").strip().lower()
        self.router_model = os.getenv("ROUTER_MODEL", self.gemini_model)

        self.openai = OpenAIProvider(api_key=self.openai_api_key) if self.openai_api_key else None
        self.gemini = GeminiProvider(api_key=self.gemini_api_key, default_model=self.gemini_model) if self.gemini_api_key else None
        self.openrouter = OpenRouterProvider(api_key=self.openrouter_api_key) if self.openrouter_api_key else None
        self.ollama = OllamaProvider(base_url=self.ollama_base_url, default_model=self.ollama_model)

    def _model_for(self, provider_name: str) -> str:
        return {
            "gemini": self.gemini_model,
            "openai": self.openai_model,
            "openrouter": self.openrouter_model,
            "ollama": self.ollama_model,
        }.get(provider_name, "unknown")

    def _rule_route(self, user_text: str) -> str:
        return choose_route(user_text).provider

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
        def _add(chain: list[tuple[str, object]], name: str, instance: object | None) -> None:
            if instance is not None:
                chain.append((name, instance))

        chain: list[tuple[str, object]] = []
        if preferred == "ollama":
            _add(chain, "ollama", self.ollama)
            _add(chain, "gemini", self.gemini)
            _add(chain, "openai", self.openai)
            _add(chain, "openrouter", self.openrouter)
        elif preferred == "gemini":
            _add(chain, "gemini", self.gemini)
            _add(chain, "openai", self.openai)
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "ollama", self.ollama)
        elif preferred in ("openai", "openrouter"):
            _add(chain, "openai", self.openai)
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "gemini", self.gemini)
            _add(chain, "ollama", self.ollama)
        else:
            _add(chain, "ollama", self.ollama)
            _add(chain, "gemini", self.gemini)
            _add(chain, "openai", self.openai)
            _add(chain, "openrouter", self.openrouter)
        return chain

    def _with_system_prompt(self, canonical: CanonicalChatRequest) -> CanonicalChatRequest:
        tenant_type = canonical.metadata.get("business_type")
        tenant_name = canonical.metadata.get("business_name", "")
        system_prompt = get_prompt(tenant_type, tenant_name)
        messages = list(canonical.messages)
        if messages and messages[0].role == "system":
            messages[0] = CanonicalMessage(role="system", content=system_prompt)
        else:
            messages = [CanonicalMessage(role="system", content=system_prompt), *messages]
        return canonical.model_copy(
            update={
                "messages": messages,
            }
        )

    def _system_prompt_for(self, canonical: CanonicalChatRequest) -> str:
        tenant_type = canonical.metadata.get("business_type")
        tenant_name = canonical.metadata.get("business_name", "")
        return get_prompt(tenant_type, tenant_name)

    async def call_with_timeout(self, provider_func, timeout_ms: int, *args, **kwargs):
        try:
            return await asyncio.wait_for(
                provider_func(*args, **kwargs),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("[TIMEOUT] provider timeout after %sms", timeout_ms)
            raise TimeoutError(f"Provider timeout {timeout_ms}ms") from exc

    def _get_redis_safe(self):
        try:
            return get_redis()
        except Exception:
            return None

    async def is_provider_healthy(self, provider: str) -> bool:
        redis_client = self._get_redis_safe()
        if redis_client is None:
            return True
        errors = await redis_client.get(f"cb:{provider}:errors")
        if errors and int(errors) >= CIRCUIT_BREAKER_THRESHOLD:
            banned_until = await redis_client.get(f"cb:{provider}:banned_until")
            if banned_until and float(banned_until) > datetime.now().timestamp():
                logger.info("[CIRCUIT BREAKER] skipping %s, banned", provider)
                return False
        return True

    async def record_provider_error(self, provider: str):
        redis_client = self._get_redis_safe()
        if redis_client is None:
            return
        key = f"cb:{provider}:errors"
        errors = await redis_client.incr(key)
        await redis_client.expire(key, 300)
        if errors >= CIRCUIT_BREAKER_THRESHOLD:
            banned_until = datetime.now() + timedelta(seconds=CIRCUIT_BREAKER_TIMEOUT)
            await redis_client.set(
                f"cb:{provider}:banned_until",
                banned_until.timestamp(),
                ex=CIRCUIT_BREAKER_TIMEOUT,
            )
            logger.warning("[CIRCUIT BREAKER] %s banned for %ss", provider, CIRCUIT_BREAKER_TIMEOUT)

    async def record_provider_success(self, provider: str):
        redis_client = self._get_redis_safe()
        if redis_client is None:
            return
        await redis_client.delete(f"cb:{provider}:errors")

    async def _call_provider(
        self,
        provider_name: str,
        provider: object,
        user_text: str,
        timeout_s: float,
        model_override: str | None = None,
    ) -> str:
        if not await self.is_provider_healthy(provider_name):
            raise RuntimeError(f"Provider {provider_name} circuit open")
        timeout_ms = max(1, int(timeout_s * 1000))
        model = model_override or self._model_for(provider_name)
        canonical = getattr(self, "_provider_canonical", None)
        if canonical is None:
            canonical = CanonicalChatRequest.from_text(user_text)
        chat_method = getattr(provider, "chat", None)
        if chat_method is not None and inspect.iscoroutinefunction(chat_method):
            kwargs: dict[str, Any] = {}
            if provider_name == "ollama":
                kwargs["timeout_s"] = timeout_s
            try:
                response = await self.call_with_timeout(chat_method, timeout_ms, canonical, model=model, **kwargs)
                await self.record_provider_success(provider_name)
                return response
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    await self.record_provider_error(provider_name)
                raise
        if provider_name == "ollama":
            try:
                response = await self.call_with_timeout(
                    provider.get_completion,
                    timeout_ms,
                    user_text,
                    model=model,
                    timeout_s=timeout_s,
                )
                await self.record_provider_success(provider_name)
                return response
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    await self.record_provider_error(provider_name)
                raise
        try:
            response = await self.call_with_timeout(provider.get_completion, timeout_ms, user_text, model=model)
            await self.record_provider_success(provider_name)
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                await self.record_provider_error(provider_name)
            raise

    def _to_canonical(self, payload: CanonicalChatRequest | ChatRequest | str) -> CanonicalChatRequest:
        if isinstance(payload, CanonicalChatRequest):
            return payload
        if isinstance(payload, ChatRequest):
            return CanonicalChatRequest.from_legacy_chat(payload)
        return CanonicalChatRequest.from_text(str(payload))

    async def _resolve_route(self, canonical: CanonicalChatRequest) -> tuple[str, str]:
        user_text = canonical.user_text
        preferred = (canonical.preferred_provider or "auto").strip().lower()
        route_reason = "preferred_provider_override"
        if preferred == "auto":
            if self.router_mode == "agent":
                preferred = await self._agent_route(user_text)
                route_reason = "agent_router_decision"
            else:
                preferred = self._rule_route(user_text)
                route_reason = choose_route(
                    user_text,
                    local_model=self.ollama_model,
                    gemini_model=self.gemini_model,
                    default_model=self.openai_model,
                ).route_reason
        return preferred, route_reason

    async def get_response(self, payload: CanonicalChatRequest | ChatRequest | str) -> dict:
        start = time.monotonic()
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        canonical = self._to_canonical(payload)
        user_text = canonical.user_text
        preferred, route_reason = await self._resolve_route(canonical)
        system_prompt = self._system_prompt_for(canonical)
        logger.info(
            "[ROUTER] type=%s user_len=%s decision=%s provider=%s reason=%s",
            canonical.metadata.get("business_type"),
            len(canonical.messages[-1].content) if canonical.messages else 0,
            self._model_for(preferred),
            preferred,
            route_reason,
        )
        logger.info("[PROMPT] system_prompt_start=%s...", system_prompt[:60])
        canonical_with_system = self._with_system_prompt(canonical)
        timeout_ms = canonical.timeout_ms or 30000
        timeout_s = max(0.001, timeout_ms / 1000.0)

        chain = self._get_provider_chain(preferred)
        if not chain:
            raise HTTPException(status_code=500, detail="No providers available")

        cache_provider = chain[0][0]
        cache_model = self._model_for(cache_provider)
        try:
            cached = await get_cached_response(user_text, cache_provider, cache_model)
        except Exception:
            cached = None

        if cached is not None:
            latency_ms = int((time.monotonic() - start) * 1000)
            prov = cached["provider"]
            mdl = cached["model"]
            prompt_tokens, completion_tokens, _ = estimate_cost(prov, mdl, user_text, cached["answer"])
            try:
                await log_request(
                    request_id=request_id,
                    timestamp=timestamp,
                    message=user_text,
                    provider=prov,
                    model=mdl,
                    latency_ms=latency_ms,
                    cached=True,
                    fallback_used=cached["fallback_used"],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=0.0,
                    status="ok",
                )
            except Exception:
                pass
            return {
                "answer": cached["answer"],
                "provider": prov,
                "model": mdl,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "request_id": request_id,
                "fallback_used": cached["fallback_used"],
                "cached": True,
                "cost_usd": 0.0,
                "route_reason": route_reason,
                "timeout_hit": False,
                "provider_chain": [cache_provider],
            }

        answer = None
        provider_name = None
        provider_model_used: str | None = None
        fallback_used = False
        timeout_hit = False
        provider_chain: list[str] = []
        last_error: Exception | None = None
        self._provider_canonical = canonical_with_system
        try:
            use_local_fast_fallback = canonical.max_cost_tier in {"low", "medium", "high"}
            if use_local_fast_fallback:
                try:
                    # Local models need 3-5s for cold start (load 4.7GB into RAM)
                    # Local models: ministral-3:3b (~3GB) cold start 15-25s, warm 0.8-1.5s
                    # llama3:8b (~4.7GB) cold start 20-30s, warm 1.2-2s
                    local_timeout_ms = min(timeout_ms, 35000)
                    local_timeout_s = max(0.001, local_timeout_ms / 1000.0)
                    local_model = "ministral-3:3b" if canonical.max_cost_tier == "low" else "llama3:8b"
                    logger.info("[LOCAL] attempting ollama/%s with timeout=%sms", local_model, local_timeout_ms)
                    provider_chain.append("ollama")
                    answer = await self._call_provider(
                        "ollama",
                        self.ollama,
                        user_text,
                        local_timeout_s,
                        model_override=local_model,
                    )
                    provider_name = "ollama"
                    provider_model_used = local_model
                    logger.info("[LOCAL SUCCESS] latency=%sms", int((time.monotonic() - start) * 1000))
                except Exception as exc:
                    logger.warning("[LOCAL FAIL] %s: %s", type(exc).__name__, str(exc)[:100])

            for i, (prov_name, prov) in enumerate(chain):
                if answer is not None:
                    break
                if use_local_fast_fallback and prov_name == "ollama":
                    continue
                for attempt in range(_MAX_RETRIES):
                    try:
                        provider_chain.append(prov_name)
                        answer = await self._call_provider(prov_name, prov, user_text, timeout_s)
                        provider_name = prov_name
                        provider_model_used = self._model_for(prov_name)
                        fallback_used = i > 0
                        break
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 404 and prov_name == "ollama":
                            model_name = self._model_for("ollama")
                            raise HTTPException(
                                status_code=400,
                                detail=f"Ollama model '{model_name}' not found. Run: ollama pull {model_name}",
                            ) from exc
                        if exc.response.status_code == 429:
                            logger.info("[FALLBACK] 429 from %s, switching", prov_name)
                            last_error = exc
                            break
                        last_error = exc
                    except TimeoutError as exc:
                        timeout_hit = True
                        logger.info("[FALLBACK] %s failed, trying next", prov_name)
                        last_error = exc
                        break
                    except Exception as exc:
                        logger.info("[FALLBACK] %s failed, trying next", prov_name)
                        last_error = exc

                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(_RETRY_DELAY_S * (attempt + 1))
                if answer is not None:
                    break
        finally:
            self._provider_canonical = None

        latency_ms = int((time.monotonic() - start) * 1000)
        model_used = provider_model_used or (self._model_for(provider_name) if provider_name else "unknown")

        if answer is None:
            error_msg = str(last_error) if last_error else "All providers failed"
            try:
                await log_request(
                    request_id=request_id,
                    timestamp=timestamp,
                    message=user_text,
                    provider=chain[0][0],
                    model=self._model_for(chain[0][0]),
                    latency_ms=latency_ms,
                    cached=False,
                    fallback_used=False,
                    prompt_tokens=None,
                    completion_tokens=None,
                    cost_usd=None,
                    status="error",
                    error=error_msg,
                )
            except Exception:
                pass
            raise HTTPException(status_code=502, detail=f"All providers failed: {error_msg}")

        prompt_tokens, completion_tokens, cost_usd = estimate_cost(provider_name, model_used, user_text, answer)
        if ":free" in model_used:
            cost_usd = 0.0

        try:
            await set_cached_response(
                user_text,
                provider_name,
                model_used,
                {
                    "answer": answer,
                    "provider": provider_name,
                    "model": model_used,
                    "fallback_used": fallback_used,
                },
            )
        except Exception:
            pass

        try:
            await log_request(
                request_id=request_id,
                timestamp=timestamp,
                message=user_text,
                provider=provider_name,
                model=model_used,
                latency_ms=latency_ms,
                cached=False,
                fallback_used=fallback_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                status="ok",
            )
        except Exception:
            pass

        return {
            "answer": answer,
            "provider": provider_name,
            "model": model_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "request_id": request_id,
            "fallback_used": fallback_used,
            "cached": False,
            "cost_usd": cost_usd,
            "route_reason": route_reason,
            "timeout_hit": timeout_hit,
            "provider_chain": provider_chain,
        }

    async def stream_response(self, payload: CanonicalChatRequest | ChatRequest | str) -> AsyncIterator[dict[str, str]]:
        canonical = self._to_canonical(payload)
        preferred, route_reason = await self._resolve_route(canonical)
        system_prompt = self._system_prompt_for(canonical)
        logger.info(
            "[ROUTER] type=%s user_len=%s decision=%s provider=%s reason=%s",
            canonical.metadata.get("business_type"),
            len(canonical.messages[-1].content) if canonical.messages else 0,
            self._model_for(preferred),
            preferred,
            route_reason,
        )
        logger.info("[PROMPT] system_prompt_start=%s...", system_prompt[:60])
        canonical_with_system = self._with_system_prompt(canonical)
        timeout_ms = canonical.timeout_ms or 30000
        timeout_s = max(1, int(timeout_ms / 1000))
        chain = self._get_provider_chain(preferred)
        if not chain:
            raise HTTPException(status_code=500, detail="No providers available")

        last_error: Exception | None = None
        for i, (provider_name, provider) in enumerate(chain):
            stream_method = getattr(provider, "stream", None)
            if stream_method is None:
                continue
            try:
                kwargs: dict[str, Any] = {}
                if provider_name == "ollama":
                    kwargs["timeout_s"] = timeout_s
                async for chunk in stream_method(canonical_with_system, model=self._model_for(provider_name), **kwargs):
                    if chunk:
                        yield {
                            "content": chunk,
                            "provider": provider_name,
                            "model": self._model_for(provider_name),
                            "route_reason": route_reason,
                            "fallback_used": i > 0,
                        }
                return
            except Exception as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                    logger.info("[FALLBACK] 429 from %s, switching", provider_name)
                if isinstance(exc, TimeoutError):
                    logger.info("[FALLBACK] timeout, switching provider")
                last_error = exc
                continue
        error_msg = str(last_error) if last_error else "All providers failed"
        raise HTTPException(status_code=502, detail=f"All providers failed: {error_msg}")
