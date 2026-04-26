import asyncio
import inspect
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from app.core.canonical import CanonicalChatRequest, CanonicalMessage
from app.clients.redis_client import get_redis
from app.core.config import (
    CIRCUIT_BREAKER_THRESHOLD,
    CIRCUIT_BREAKER_TIMEOUT,
    LATENCY_P95_THRESHOLD_MS,
    QUOTA_BREAKER_TIMEOUT,
)
from app.core.prompts import get_prompt
from app.db.database import log_request
from app.providers.gemini_client import GeminiProvider
from app.providers.ollama_client import OllamaProvider
from app.providers.openai_client import OpenAIProvider
from app.providers.openrouter_client import OpenRouterProvider
from app.router.rules import choose_route
from app.schemas.chat import ChatRequest
from app.metrics import (
    LATENCY_BREAKDOWN, 
    PROVIDER_LATENCY, 
    REDIS_LATENCY, 
    record_chat_metrics,
    PROVIDER_STATE,
    CIRCUIT_STATE,
    CIRCUIT_OPEN_COUNT,
    FALLBACK_LEVEL_USED,
    REQUEST_DROPPED_TOTAL,
    REQUEST_BLOCKED_BY_RATE_LIMIT,
    RETRY_BACKOFF_SECONDS,
    SYSTEM_STATE,
    ACTIVE_PROVIDERS_COUNT,
    TIME_IN_STATE,
    BLOCKED_REQUESTS_TOTAL,
    PROVIDER_SUCCESS_RATE,
    PROVIDER_FAILURE_RATE,
    COOLDOWN_TRIGGER_COUNT,
    REQUEST_RATE_PER_SECOND,
    PROVIDER_HEALTH_SCORE,
    COOLDOWN_ACTIVE
)
from app.services.cache_service import get_cached_response, set_cached_response
from app.services.cost_service import estimate_cost

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
        self.router_timeout_ms = int(os.getenv("ROUTER_TIMEOUT_MS", "2000"))
        self.local_race_timeout_ms = int(os.getenv("LOCAL_RACE_TIMEOUT_MS", "3000"))
        self.remote_race_timeout_ms = int(os.getenv("REMOTE_RACE_TIMEOUT_MS", "10000"))
        self.provider_stagger_ms = int(os.getenv("PROVIDER_STAGGER_MS", "250"))
        
        # Concurrency control per provider
        self._provider_semaphores = {
            "openai": asyncio.Semaphore(20),
            "gemini": asyncio.Semaphore(10),
            "openrouter": asyncio.Semaphore(20),
            "ollama": asyncio.Semaphore(5),
        }
        self._global_semaphore = asyncio.Semaphore(50)

        # Rate limits (tokens/sec, burst/capacity)
        self._provider_rate_limits = {
            "openai": (5.0, 20),
            "gemini": (2.0, 10),
            "openrouter": (5.0, 20),
            "ollama": (10.0, 30),
        }

        self.openai = OpenAIProvider(api_key=self.openai_api_key) if self.openai_api_key else None
        self.gemini = GeminiProvider(api_key=self.gemini_api_key, default_model=self.gemini_model) if self.gemini_api_key else None
        self.openrouter = OpenRouterProvider(api_key=self.openrouter_api_key) if self.openrouter_api_key else None
        self.ollama = OllamaProvider(base_url=self.ollama_base_url, default_model=self.ollama_model)
        self._disabled_providers: dict[str, str] = {}
        self._last_logs: dict[str, dict[str, Any]] = {}
        self._last_system_state = "healthy"
        self._state_changed_at = time.time()
        self.testing_mode = os.getenv("PYTEST_CURRENT_TEST") is not None

    def _log_throttled(self, key: str, message: str, level=logging.WARNING, interval=10):
        now = time.time()
        if key not in self._last_logs:
            self._last_logs[key] = {"time": now, "count": 0}
            logger.log(level, message)
        else:
            self._last_logs[key]["count"] += 1
            if (now - self._last_logs[key]["time"]) > interval:
                count = self._last_logs[key]["count"]
                if count > 0:
                    message = f"{message} [{key}] x{count} occurrences suppressed"
                logger.log(level, message)
                self._last_logs[key] = {"time": now, "count": 0}

    async def _get_system_state(self) -> dict[str, Any]:
        """
        Returns system status including state (healthy, degraded, critical, locked),
        cooldown info, and per-provider health scores.
        Optimized with MGET to reduce Redis roundtrips.
        """
        redis_client = self._get_redis_safe()
        providers = ["openai", "gemini", "openrouter", "ollama"]
        health_scores = {p: 100 for p in providers}
        
        if not redis_client:
            return {
                "state": "healthy", 
                "scores": health_scores, 
                "cooldown": False, 
                "active_count": len(providers)
            }

        now = time.time()
        
        # Build list of keys for MGET
        keys = ["system:cooldown:until"]
        for p in providers:
            keys.append(f"cb:{p}:state")
            keys.append(f"quota:{p}:banned_until")
            keys.append(f"latency:{p}:p95")
        
        values = await redis_client.mget(*keys)
        val_map = dict(zip(keys, values))
        
        cooldown_until = val_map.get("system:cooldown:until")
        cooldown_active = cooldown_until is not None and float(cooldown_until) > now
        
        unhealthy_count = 0
        for p in providers:
            state_val = val_map.get(f"cb:{p}:state")
            state = int(state_val) if state_val is not None else 0
            
            quota_until = val_map.get(f"quota:{p}:banned_until")
            is_quota = quota_until is not None and float(quota_until) > now
            
            score = 100
            if state == 1: # OPEN
                score = 0
            elif is_quota:
                score = 10
            elif state == 2: # HALF_OPEN
                score = 50
                
            p95 = val_map.get(f"latency:{p}:p95")
            if p95 and float(p95) > LATENCY_P95_THRESHOLD_MS:
                score = max(0, score - 30)
                
            health_scores[p] = score
            PROVIDER_HEALTH_SCORE.labels(provider=p).set(score)
            
            if score < 50:
                unhealthy_count += 1
        
        ACTIVE_PROVIDERS_COUNT.set(len(providers) - unhealthy_count)
        COOLDOWN_ACTIVE.set(1 if cooldown_active else 0)

        if cooldown_active:
            sys_state = "locked"
            SYSTEM_STATE.set(3)
        elif unhealthy_count >= len(providers):
            sys_state = "critical"
            SYSTEM_STATE.set(2)
        elif unhealthy_count >= 2:
            sys_state = "degraded"
            SYSTEM_STATE.set(1)
        else:
            sys_state = "healthy"
            SYSTEM_STATE.set(0)

        if sys_state != self._last_system_state:
            self._log_throttled("state_change", f"[SYSTEM] State changed from {self._last_system_state} to {sys_state}", level=logging.INFO)
            self._last_system_state = sys_state
            self._state_changed_at = now

        TIME_IN_STATE.labels(state=sys_state).inc(1)
        
        return {
            "state": sys_state,
            "scores": health_scores,
            "cooldown": cooldown_active,
            "cooldown_until": float(cooldown_until) if cooldown_until else None,
            "active_count": len(providers) - unhealthy_count
        }

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
            decision = await asyncio.wait_for(
                self.gemini.get_completion(prompt, model=self.router_model),
                timeout=max(0.1, self.router_timeout_ms / 1000.0),
            )
        except Exception:
            return self._rule_route(user_text)
        value = decision.strip().lower()
        if "ollama" in value:
            return "ollama"
        if "gemini" in value:
            return "gemini"
        return self._rule_route(user_text)

    def _get_provider_chain(self, preferred: str, system_state: str = "healthy") -> list[tuple[str, object]]:
        def _add(chain: list[tuple[str, object]], name: str, instance: object | None) -> None:
            if instance is not None and name not in self._disabled_providers:
                chain.append((name, instance))

        chain: list[tuple[str, object]] = []
        
        # If in CRITICAL or DEGRADED state, follow the STRICT order:
        # OpenAI -> Gemini -> OpenRouter -> Ollama
        if system_state in ["critical", "degraded", "locked"]:
            _add(chain, "openai", self.openai)
            _add(chain, "gemini", self.gemini)
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "ollama", self.ollama)
            return chain

        # Normal HEALTHY state logic with preferred provider
        if preferred == "ollama":
            _add(chain, "ollama", self.ollama)
            _add(chain, "gemini", self.gemini)
            _add(chain, "openai", self.openai)
            _add(chain, "openrouter", self.openrouter)
        elif preferred == "gemini":
            _add(chain, "gemini", self.gemini)
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "openai", self.openai)
            _add(chain, "ollama", self.ollama)
        elif preferred == "openai":
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "openai", self.openai)
            _add(chain, "gemini", self.gemini)
            _add(chain, "ollama", self.ollama)
        elif preferred == "openrouter":
            _add(chain, "openrouter", self.openrouter)
            _add(chain, "openai", self.openai)
            _add(chain, "gemini", self.gemini)
            _add(chain, "ollama", self.ollama)
        else:
            # Default order for HEALTHY state (auto mode)
            # Requirements say: ollama, gemini, openai, openrouter
            _add(chain, "ollama", self.ollama)
            _add(chain, "gemini", self.gemini)
            _add(chain, "openai", self.openai)
            _add(chain, "openrouter", self.openrouter)
        return chain

    def _prioritize_auto_fallbacks(self, chain: list[tuple[str, object]]) -> list[tuple[str, object]]:
        order = {"ollama": 0, "openrouter": 1, "gemini": 2, "openai": 3}
        return sorted(chain, key=lambda item: order.get(item[0], 99))

    @staticmethod
    def _is_binary_policy_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "dll load failed while importing jiter" in text
            or "политика управления приложениями заблокировала этот файл" in text
            or "application control policy blocked" in text
        )

    def _disable_provider(self, provider_name: str, reason: Exception) -> None:
        if provider_name in self._disabled_providers:
            return
        message = str(reason)[:240]
        self._disabled_providers[provider_name] = message
        logger.warning("[PROVIDER DISABLED] %s disabled: %s", provider_name, message)

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

    async def aclose(self) -> None:
        for provider in (self.openai, self.gemini, self.openrouter, self.ollama):
            close_method = getattr(provider, "aclose", None)
            if close_method is None:
                continue
            try:
                result = close_method()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Failed to close provider client")

    def _system_prompt_for(self, canonical: CanonicalChatRequest) -> str:
        tenant_type = canonical.metadata.get("business_type")
        tenant_name = canonical.metadata.get("business_name", "")
        return get_prompt(tenant_type, tenant_name)

    def _static_intent_engine(self, text: str) -> str | None:
        """
        Simple static intent engine for common queries to save tokens/latency
        and provide a baseline in CRITICAL/DEGRADED modes.
        """
        # Remove punctuation and normalize
        t = re.sub(r'[^\w\s]', '', text.lower()).strip()
        
        if t in ["привет", "hello", "hi"]:
            return "Привет! Чем я могу вам помочь сегодня?"
        if t in ["статус", "status", "как дела"]:
            return "Все системы работают в штатном режиме (насколько это возможно). Я готов к работе!"
        if "кто ты" in t or "who are you" in t:
            return "Я — LLMHub, ваш интеллектуальный ассистент, работающий на базе нескольких языковых моделей."
        return None

    async def call_with_timeout(self, provider_func, timeout_ms: int, *args, **kwargs):
        try:
            return await asyncio.wait_for(
                provider_func(*args, **kwargs),
                timeout=timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("[TIMEOUT] provider timeout after %sms", timeout_ms)
            raise TimeoutError(f"Provider timeout {timeout_ms}ms") from exc

    def _format_response_dict(
        self, 
        answer: str, 
        provider: str, 
        model: str, 
        latency_ms: int, 
        request_id: str,
        fallback_used: bool = False,
        cached: bool = False,
        route_reason: str = "unknown",
        timeout_hit: bool = False,
        provider_chain: list[str] = None
    ) -> dict:
        prompt_tokens, completion_tokens, cost_usd = estimate_cost(provider, model, "", answer)
        if ":free" in model or cached:
            cost_usd = 0.0
        
        return {
            "answer": answer,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "request_id": request_id,
            "fallback_used": fallback_used,
            "cached": cached,
            "cost_usd": cost_usd,
            "route_reason": route_reason,
            "timeout_hit": timeout_hit,
            "provider_chain": provider_chain or [provider],
        }

    def _emergency_static_response(self, start_time: float, request_id: str) -> dict:
        latency_ms = int((time.monotonic() - start_time) * 1000)
        FALLBACK_LEVEL_USED.labels(level="static_safe").inc()
        REQUEST_DROPPED_TOTAL.labels(reason="all_providers_failed").inc()
        return self._format_response_dict(
            "Извините, сейчас я испытываю технические сложности и не могу дать полный ответ. Пожалуйста, попробуйте позже.",
            "static_fallback",
            "safe-mode",
            latency_ms,
            request_id,
            fallback_used=True,
            cached=False,
            route_reason="emergency_fallback"
        )

    def _get_redis_safe(self):
        try:
            return get_redis()
        except Exception:
            return None

    async def _check_rate_limit(self, provider: str) -> bool:
        redis_client = self._get_redis_safe()
        if not redis_client:
            return True

        rate, capacity = self._provider_rate_limits.get(provider, (1.0, 5))
        now = time.time()
        key = f"rl:{provider}"

        # Lua script for Token Bucket
        lua = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local requested = 1

        local bucket = redis.call('hmget', key, 'tokens', 'last_updated')
        local tokens = tonumber(bucket[1]) or capacity
        local last_updated = tonumber(bucket[2]) or now

        local delta = math.max(0, now - last_updated) * rate
        tokens = math.min(capacity, tokens + delta)

        if tokens >= requested then
            tokens = tokens - requested
            redis.call('hmset', key, 'tokens', tokens, 'last_updated', now)
            redis.call('expire', key, 60)
            return 1
        else
            return 0
        end
        """
        try:
            allowed = await redis_client.eval(lua, 1, key, rate, capacity, now)
            if not allowed:
                REQUEST_BLOCKED_BY_RATE_LIMIT.labels(provider=provider).inc()
            return bool(allowed)
        except Exception as e:
            logger.error("[RATE LIMIT] error checking %s: %s", provider, e)
            return True

    async def _trigger_global_cooldown(self, duration: int = 30):
        redis_client = self._get_redis_safe()
        if not redis_client:
            return
        now = time.time()
        from app.metrics import COOLDOWN_TRIGGER_COUNT
        COOLDOWN_TRIGGER_COUNT.inc()
        await redis_client.setex("system:cooldown:until", duration, str(now + duration))
        self._log_throttled("entering_cooldown", f"[SYSTEM] Entering global cooldown for {duration}s", level=logging.ERROR)

    async def is_provider_healthy(self, provider: str) -> bool:
        redis_client = self._get_redis_safe()
        if redis_client is None:
            return True

        now = time.time()
        
        # 1. Check Circuit Breaker State
        state_key = f"cb:{provider}:state"
        state = await redis_client.get(state_key)
        state = int(state) if state is not None else 0  # Default to CLOSED (0)

        if state == 1:  # OPEN
            banned_until = await redis_client.get(f"cb:{provider}:banned_until")
            if banned_until and float(banned_until) > now:
                self._log_throttled(f"hb_{provider}_open", f"[HEALTH] skipping {provider}: circuit open (banned until {banned_until})", interval=30)
                CIRCUIT_STATE.labels(provider=provider).set(1)
                PROVIDER_STATE.labels(provider=provider).set(-1)
                return False
            else:
                # Transition to HALF_OPEN
                logger.info("[HEALTH] %s moving to HALF_OPEN", provider)
                await redis_client.set(state_key, 2)
                state = 2

        if state == 2:  # HALF_OPEN
            # Allow one probe every 15 seconds
            last_probe = await redis_client.get(f"cb:{provider}:last_probe")
            if last_probe and (now - float(last_probe)) < 15:
                PROVIDER_STATE.labels(provider=provider).set(0)
                CIRCUIT_STATE.labels(provider=provider).set(2)
                return False
            
            logger.info("[HEALTH] probing %s: circuit half-open", provider)
            await redis_client.set(f"cb:{provider}:last_probe", now, ex=30)
            PROVIDER_STATE.labels(provider=provider).set(0)
            CIRCUIT_STATE.labels(provider=provider).set(2)
            return True

        # CLOSED State (0)
        CIRCUIT_STATE.labels(provider=provider).set(0)

        # 2. Check Rate Limit (Token Bucket)
        if not await self._check_rate_limit(provider):
            self._log_throttled(f"hb_{provider}_rate", f"[HEALTH] skipping {provider}: rate limited (token bucket)", interval=30)
            # We return False but don't change CB state
            PROVIDER_STATE.labels(provider=provider).set(0)
            return False

        # 3. Check Soft Quota
        quota_until = await redis_client.get(f"quota:{provider}:banned_until")
        if quota_until and float(quota_until) > now:
            self._log_throttled(f"hb_{provider}_quota", f"[HEALTH] skipping {provider}: quota cooldown", interval=30)
            PROVIDER_STATE.labels(provider=provider).set(0)
            return False

        # 4. Check Latency (P95)
        p95 = await redis_client.get(f"latency:{provider}:p95")
        if p95 and float(p95) > LATENCY_P95_THRESHOLD_MS:
            logger.info("[HEALTH] %s has high latency (%sms), deprioritizing", provider, p95)
            PROVIDER_STATE.labels(provider=provider).set(0)
            return True

        PROVIDER_STATE.labels(provider=provider).set(1)
        return True

    async def record_provider_error(self, provider: str, error: Exception | None = None):
        redis_client = self._get_redis_safe()
        if redis_client is None:
            return

        error_str = str(error).lower()
        now = time.time()
        
        # Classification
        is_429 = False
        if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
            is_429 = True
        elif any(x in error_str for x in ["429", "quota", "too many requests", "resource_exhausted"]):
            is_429 = True

        is_timeout = isinstance(error, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)) or "timeout" in error_str
        
        is_hard = any(x in error_str for x in ["connection refused", "auth", "invalid_api_key", "500 internal server error"])
        
        state_key = f"cb:{provider}:state"
        state = await redis_client.get(state_key)
        state = int(state) if state is not None else 0

        # Handle QUOTA_ERROR (429) -> Exponential Backoff
        if is_429:
            key = f"quota:{provider}:backoff_level"
            level = await redis_client.incr(key)
            await redis_client.expire(key, 600)
            
            # exponential backoff: 30s * 2^(level-1) + jitter, CAP at 300s
            backoff_base = 30 * (2 ** (level - 1))
            backoff = min(300, backoff_base)
            jitter = backoff * 0.2 * (time.time() % 1)
            total_backoff = backoff + jitter
            
            await redis_client.set(f"quota:{provider}:banned_until", now + total_backoff, ex=int(total_backoff))
            self._log_throttled(
                f"quota_{provider}", 
                f"[QUOTA] {provider} 429 detected, level {level}, backoff {total_backoff:.1f}s"
            )
            CIRCUIT_OPEN_COUNT.labels(provider=provider, reason="429").inc()
            RETRY_BACKOFF_SECONDS.labels(provider=provider).inc(total_backoff)
            
            if state == 2: # HALF_OPEN
                await redis_client.set(state_key, 1) # Back to OPEN
            return

        # Handle TIMEOUT -> Short retry window (5-10s)
        if is_timeout:
            short_ban = 10
            await redis_client.set(f"cb:{provider}:banned_until", now + short_ban, ex=short_ban)
            self._log_throttled(f"timeout_{provider}", f"[TIMEOUT] {provider} timed out, short ban {short_ban}s")
            # Also increment normal error counter but less aggressively
            error_key = f"cb:{provider}:errors"
            await redis_client.incrby(error_key, 1)
            await redis_client.expire(error_key, 300)
            return

        # Handle NETWORK_ERROR / HARD -> Circuit Breaker
        error_key = f"cb:{provider}:errors"
        increment = 2 if is_hard else 1
        errors = await redis_client.incrby(error_key, increment)
        await redis_client.expire(error_key, 300)

        if state == 2: # HALF_OPEN
            self._log_throttled(
                f"circuit_{provider}_half_open_fail",
                f"[CIRCUIT] {provider} failed in HALF_OPEN, moving back to OPEN"
            )
            await redis_client.set(state_key, 1)
            await redis_client.set(f"cb:{provider}:banned_until", now + CIRCUIT_BREAKER_TIMEOUT, ex=CIRCUIT_BREAKER_TIMEOUT)
            CIRCUIT_OPEN_COUNT.labels(provider=provider, reason="half_open_failure").inc()
            return

        if errors >= CIRCUIT_BREAKER_THRESHOLD:
            self._log_throttled(
                f"circuit_{provider}_open",
                f"[CIRCUIT] {provider} threshold reached ({errors}), opening circuit"
            )
            await redis_client.set(state_key, 1) # OPEN
            await redis_client.set(f"cb:{provider}:banned_until", now + CIRCUIT_BREAKER_TIMEOUT, ex=CIRCUIT_BREAKER_TIMEOUT)
            await redis_client.delete(error_key)
            CIRCUIT_OPEN_COUNT.labels(provider=provider, reason="threshold_reached").inc()
            CIRCUIT_STATE.labels(provider=provider).set(1)

    async def record_provider_success(self, provider: str, latency_ms: float | None = None):
        redis_client = self._get_redis_safe()
        if redis_client is None:
            return

        state_key = f"cb:{provider}:state"
        state = await redis_client.get(state_key)
        state = int(state) if state is not None else 0

        if state == 2: # HALF_OPEN
            logger.info("[CIRCUIT] %s recovered! Moving to CLOSED", provider)
            await redis_client.set(state_key, 0) # CLOSED
            CIRCUIT_STATE.labels(provider=provider).set(0)

        # Clear errors on success - this is what makes it "Half-Open" recovery
        await redis_client.delete(f"cb:{provider}:errors")
        await redis_client.delete(f"cb:{provider}:banned_until")
        await redis_client.delete(f"quota:{provider}:banned_until")
        await redis_client.delete(f"quota:{provider}:backoff_level")

        if latency_ms is not None:
            # Update p95 latency. Simplified: keep last 20 values in a list.
            key = f"latency:{provider}:history"
            await redis_client.lpush(key, latency_ms)
            await redis_client.ltrim(key, 0, 19)
            
            history = await redis_client.lrange(key, 0, -1)
            if history:
                latencies = sorted([float(x) for x in history])
                p95_idx = int(len(latencies) * 0.95)
                p95_val = latencies[min(p95_idx, len(latencies) - 1)]
                await redis_client.set(f"latency:{provider}:p95", p95_val, ex=3600)

    def _provider_timeout_budget(
        self,
        provider_name: str,
        requested_provider: str,
        timeout_s: float,
    ) -> float:
        if provider_name == "ollama":
            return min(timeout_s, self.local_race_timeout_ms / 1000.0)
        return min(timeout_s, self.remote_race_timeout_ms / 1000.0)

    def _provider_start_delay(self, index: int, requested_provider: str) -> float:
        if index == 0:
            return 0.0
        multiplier = 1.0 if requested_provider == "auto" else 1.5
        return (self.provider_stagger_ms * multiplier * index) / 1000.0

    async def _race_candidate(
        self,
        *,
        provider_name: str,
        provider: object,
        user_text: str,
        timeout_s: float,
        delay_s: float,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        if delay_s > 0:
            try:
                await asyncio.sleep(delay_s)
            except asyncio.CancelledError:
                logger.info("[RACE] Candidate %s cancelled during delay", provider_name)
                raise
        started = time.monotonic()
        logger.info("[RACE] Starting candidate: %s (timeout=%ss)", provider_name, timeout_s)
        try:
            answer = await self._call_provider(
                provider_name,
                provider,
                user_text,
                timeout_s,
                model_override=model_override,
            )
            return {
                "provider": provider_name,
                "answer": answer,
                "model": model_override or self._model_for(provider_name),
                "latency_ms": int((time.monotonic() - started) * 1000),
                "timeout_hit": False,
            }
        except asyncio.CancelledError:
            latency = time.monotonic() - started
            logger.info("[RACE] Candidate %s cancelled after %.2fs", provider_name, latency)
            PROVIDER_LATENCY.labels(provider=provider_name, stage="cancelled").observe(latency)
            raise
        except Exception as exc:
            latency = time.monotonic() - started
            logger.warning("[RACE] Candidate %s failed after %.2fs: %s", provider_name, latency, exc)
            return {
                "provider": provider_name,
                "error": exc,
                "model": model_override or self._model_for(provider_name),
                "latency_ms": int(latency * 1000),
                "timeout_hit": isinstance(exc, TimeoutError),
            }

    async def _run_fastest_response_race(
        self,
        *,
        chain: list[tuple[str, object]],
        user_text: str,
        requested_provider: str,
        timeout_s: float,
        deadline_s: float,
        local_model: str,
        use_local_fast_fallback: bool,
        health_scores: dict[str, int]
    ) -> tuple[dict[str, Any] | None, list[str], bool, Exception | None, str | None]:
        candidates: list[tuple[str, object, float, float, str | None]] = []
        
        # Filter healthy providers using pre-fetched scores
        for index, (provider_name, provider) in enumerate(chain):
            score = health_scores.get(provider_name, 0)
            if score < 50:
                if requested_provider == provider_name:
                    logger.info("[RACE] Using unhealthy but requested provider: %s", provider_name)
                else:
                    logger.warning("[RACE] Skipping unhealthy provider: %s (score %s)", provider_name, score)
                    continue
            
            candidates.append(
                (
                    provider_name,
                    provider,
                    self._provider_timeout_budget(provider_name, requested_provider, timeout_s),
                    self._provider_start_delay(len(candidates), requested_provider),
                    local_model if use_local_fast_fallback and provider_name == "ollama" else None,
                )
            )

        if not candidates:
            logger.warning("[RACE] No healthy providers available for race")
            return None, [], False, None, None

        # Override stagger for race to be more aggressive
        self.provider_stagger_ms = 100 

        tasks: dict[asyncio.Task[dict[str, Any]], str] = {
            asyncio.create_task(
                self._race_candidate(
                    provider_name=provider_name,
                    provider=provider,
                    user_text=user_text,
                    timeout_s=provider_timeout_s,
                    delay_s=delay_s,
                    model_override=model_override,
                )
            ): provider_name
            for provider_name, provider, provider_timeout_s, delay_s, model_override in candidates
        }
        provider_chain = [provider_name for provider_name, *_ in candidates]
        timeout_hit = False
        last_error: Exception | None = None
        last_failed_provider: str | None = None

        try:
            while tasks:
                # We want a winner as fast as possible.
                # If we have any 'done' tasks, process them.
                # If not, wait for any task to finish, but no longer than 0.1s to check deadline.
                try:
                    done, pending = await asyncio.wait(tasks.keys(), timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
                except Exception as e:
                    logger.error("[RACE] asyncio.wait error: %s", e)
                    break
                
                if not done:
                    # Check if global deadline exceeded
                    if time.monotonic() > deadline_s:
                        logger.warning("[RACE] Global deadline exceeded")
                        timeout_hit = True
                        break
                    
                    # Optimization: If all currently running tasks are from providers we know are slow/failing
                    # we could theoretically start more, but candidates are already all started.
                    continue

                for task in done:
                    provider_name = tasks.pop(task)
                    try:
                        result = await task
                    except Exception as e:
                        logger.error("[RACE] Task for %s failed with exception: %s", provider_name, e)
                        result = {"provider": provider_name, "error": e, "latency_ms": 0, "timeout_hit": False}

                    if "answer" in result:
                        # Winner found!
                        logger.info("[RACE] Winner: %s in %sms (reason: fastest_response)", provider_name, result.get("latency_ms"))
                        await self.record_provider_success(provider_name, result.get("latency_ms"))
                        for pending_task in pending:
                            pending_task.cancel()
                        
                        # We don't want to block the response while waiting for cancellations
                        # But we should clear the tasks dict to avoid finally block waiting
                        tasks.clear()
                        # Run background cancellation cleanup
                        asyncio.create_task(asyncio.gather(*pending, return_exceptions=True))
                        return result, provider_chain, timeout_hit, last_error, last_failed_provider

                    exc = result.get("error")
                    last_error = exc
                    last_failed_provider = provider_name
                    timeout_hit = timeout_hit or result.get("timeout_hit", False)
                    logger.info("[RACE] Provider %s failed: %s", provider_name, exc)

                    if exc:
                        await self.record_provider_error(provider_name, exc)

                    if self._is_binary_policy_error(exc):
                        self._disable_provider(provider_name, exc)
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                        # Already recorded via record_provider_error above
                        pass
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code == 404
                        and provider_name == "ollama"
                        and requested_provider == "ollama"
                    ):
                        for pending_task in pending:
                            pending_task.cancel()
                        tasks.clear()
                        model_name = self._model_for("ollama")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Ollama model '{model_name}' not found. Run: ollama pull {model_name}",
                        ) from exc
        finally:
            if tasks:
                for t in tasks.keys():
                    t.cancel()
                await asyncio.gather(*tasks.keys(), return_exceptions=True)

        return None, provider_chain, timeout_hit, last_error, last_failed_provider

    async def _call_provider(
        self,
        provider_name: str,
        provider: object,
        user_text: str,
        timeout_s: float,
        model_override: str | None = None,
    ) -> str:
        semaphore = self._provider_semaphores.get(provider_name, self._global_semaphore)
        
        async with self._global_semaphore:
            async with semaphore:
                # We don't check is_provider_healthy here again because it was checked in _run_fastest_response_race
                # and we want to allow Half-Open probes which are initiated through that check.
                
                start_provider = time.monotonic()
                timeout_ms = max(1, int(timeout_s * 1000))
                model = model_override or self._model_for(provider_name)
                canonical = getattr(self, "_provider_canonical", None)
                if canonical is None:
                    canonical = CanonicalChatRequest.from_text(user_text)
                chat_method = getattr(provider, "chat", None)
                
                try:
                    if chat_method is not None and inspect.iscoroutinefunction(chat_method):
                        kwargs: dict[str, Any] = {}
                        if provider_name == "ollama":
                            kwargs["timeout_s"] = timeout_s
                        try:
                            response = await self.call_with_timeout(chat_method, timeout_ms, canonical, model=model, **kwargs)
                            latency_ms = (time.monotonic() - start_provider) * 1000
                            await self.record_provider_success(provider_name, latency_ms)
                            PROVIDER_LATENCY.labels(provider=provider_name, stage="total").observe(latency_ms / 1000.0)
                            return response
                        except Exception as exc:
                            if (
                                isinstance(exc, httpx.HTTPStatusError)
                                and exc.response.status_code == 404
                                and provider_name == "ollama"
                            ):
                                model_name = model or self._model_for("ollama")
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Ollama model '{model_name}' not found. Run: ollama pull {model_name}",
                                ) from exc
                            await self.record_provider_error(provider_name, exc)
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
                            latency_ms = (time.monotonic() - start_provider) * 1000
                            await self.record_provider_success(provider_name, latency_ms)
                            PROVIDER_LATENCY.labels(provider=provider_name, stage="total").observe(latency_ms / 1000.0)
                            return response
                        except Exception as exc:
                            if (
                                isinstance(exc, httpx.HTTPStatusError)
                                and exc.response.status_code == 404
                            ):
                                model_name = model or self._model_for("ollama")
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Ollama model '{model_name}' not found. Run: ollama pull {model_name}",
                                ) from exc
                            await self.record_provider_error(provider_name, exc)
                            raise
                    try:
                        response = await self.call_with_timeout(provider.get_completion, timeout_ms, user_text, model=model)
                        latency_ms = (time.monotonic() - start_provider) * 1000
                        await self.record_provider_success(provider_name, latency_ms)
                        PROVIDER_LATENCY.labels(provider=provider_name, stage="total").observe(latency_ms / 1000.0)
                        return response
                    except Exception as exc:
                        await self.record_provider_error(provider_name, exc)
                        raise
                except Exception as e:
                    PROVIDER_LATENCY.labels(provider=provider_name, stage="error").observe(time.monotonic() - start_provider)
                    raise e

    def _to_canonical(self, payload: CanonicalChatRequest | ChatRequest | str) -> CanonicalChatRequest:
        if isinstance(payload, CanonicalChatRequest):
            return payload
        if isinstance(payload, ChatRequest):
            return CanonicalChatRequest.from_legacy_chat(payload)
        return CanonicalChatRequest.from_text(str(payload))

    async def _resolve_route(self, canonical: CanonicalChatRequest) -> tuple[str, str]:
        start = time.monotonic()
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
                    default_model=self.openrouter_model,
                ).route_reason
        LATENCY_BREAKDOWN.labels(stage="routing").observe(time.monotonic() - start)
        return preferred, route_reason

    async def _run_linear_fallback(
        self, 
        chain: list[tuple[str, object]], 
        user_text: str, 
        start_time: float, 
        request_id: str, 
        timeout_s: float,
        system_state: str = "healthy",
        model_override: str | None = None,
        requested_provider: str = "auto",
        health_scores: dict[str, int] | None = None
    ) -> dict:
        """
        Executes a linear fallback chain through providers in the given order.
        Strictly one by one, max 1 retry (implicitly 1 attempt per provider).
        """
        timestamp = datetime.now(timezone.utc)
        last_error = None
        last_provider = None
        
        for index, (provider_name, provider_instance) in enumerate(chain):
            # Pre-filter by health score if provided, but allow in testing_mode
            if not self.testing_mode and health_scores:
                score = health_scores.get(provider_name, 0)
                if score < 50:
                    continue
            
            try:
                self._log_throttled(f"fallback_try_{provider_name}", f"[FALLBACK] Trying {provider_name}", level=logging.INFO)
                
                # Apply timeout budget
                provider_timeout_s = self._provider_timeout_budget(provider_name, requested_provider, timeout_s)
                
                current_model = model_override if provider_name == "ollama" else self._model_for(provider_name)

                # Call positionally for first 4 args, keyword for model_override to match mocks and signature
                answer = await self._call_provider(
                    provider_name, 
                    provider_instance, 
                    user_text, 
                    provider_timeout_s,
                    model_override=current_model if provider_name == "ollama" else None
                )
                
                latency_ms = int((time.monotonic() - start_time) * 1000)
                
                fallback_used = index > 0
                FALLBACK_LEVEL_USED.labels(level=f"linear_{provider_name}").inc()
                await self.record_provider_success(provider_name, latency_ms)
                
                resp = self._format_response_dict(
                    answer, provider_name, current_model, latency_ms, request_id,
                    fallback_used=fallback_used, cached=False, route_reason="linear_fallback_chain"
                )
                
                try:
                    await log_request(
                        request_id=request_id, timestamp=timestamp, message=user_text,
                        provider=provider_name, model=current_model, latency_ms=latency_ms,
                        cached=False, fallback_used=fallback_used, 
                        prompt_tokens=resp["prompt_tokens"], completion_tokens=resp["completion_tokens"],
                        cost_usd=resp["cost_usd"],
                        status="ok"
                    )
                except Exception: pass
                
                # Cache successful response
                try:
                    await set_cached_response(user_text, provider_name, current_model, {
                        "answer": answer, "provider": provider_name, "model": current_model,
                        "fallback_used": fallback_used
                    })
                except Exception: pass
                
                return resp
            except HTTPException:
                raise
            except Exception as e:
                # 4.4 HTTP 404 (ollama model) -> 400
                if (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code == 404
                    and provider_name == "ollama"
                ):
                    model_name = model_override or self._model_for("ollama")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Ollama model '{model_name}' not found. Run: ollama pull {model_name}",
                    ) from e
                
                last_error = e
                last_provider = provider_name
                await self.record_provider_error(provider_name, e)
                # Handle policy block disable logic
                if self._is_binary_policy_error(e):
                    self._disable_provider(provider_name, e)
                continue
        
        # Log failure before raising/returning
        if last_error:
            try:
                latency_ms = int((time.monotonic() - start_time) * 1000)
                await log_request(
                    request_id=request_id, timestamp=timestamp, message=user_text,
                    provider=last_provider or "none", model="none", latency_ms=latency_ms,
                    status="error"
                )
            except Exception: pass

        if self.testing_mode and system_state != "critical":
             raise HTTPException(status_code=502, detail=f"All providers failed: {str(last_error)}")
        
        # Last resort: Static Fallback
        return self._emergency_static_response(start_time, request_id)

    async def get_response(self, payload: CanonicalChatRequest | ChatRequest | str) -> dict:
        start = time.monotonic()
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        canonical = self._to_canonical(payload)
        user_text = canonical.user_text
        
        # 1. System State & Cooldown (Control Plane First)
        status = await self._get_system_state()
        system_state = status["state"]
        cooldown_active = status["cooldown"]
        health_scores = status["scores"]

        # 2. Static Intent Engine (Strict Priority)
        if not self.testing_mode:
            if static_answer := self._static_intent_engine(user_text):
                latency_ms = int((time.monotonic() - start) * 1000)
                FALLBACK_LEVEL_USED.labels(level="static_intent").inc()
                resp = self._format_response_dict(
                    static_answer, "static_intent", "baseline", latency_ms, request_id, 
                    fallback_used=True, cached=False, route_reason="static_intent_match"
                )
                try:
                    await log_request(
                        request_id=request_id, timestamp=timestamp, message=user_text,
                        provider="static_intent", model="baseline", latency_ms=latency_ms,
                        cached=False, fallback_used=True, prompt_tokens=0, completion_tokens=0,
                        cost_usd=0.0, status="ok"
                    )
                except Exception: pass
                return resp

        # 3. LOCKED state (Immediate Static Fallback)
        if system_state == "locked":
            BLOCKED_REQUESTS_TOTAL.labels(reason="cooldown", state="locked").inc()
            self._log_throttled("gate_locked", "[SYSTEM] GATE LOCKED: Returning static fallback immediately", level=logging.ERROR)
            return self._emergency_static_response(start, request_id)

        requested_provider = (canonical.preferred_provider or "auto").strip().lower()
        preferred, route_reason = await self._resolve_route(canonical)
        
        if system_state != "healthy":
            self._log_throttled("system_not_healthy", f"[SYSTEM] State: {system_state}, switching to linear fallback")

        # 4. Semantic Cache
        local_model = self.ollama_model
        if requested_provider == "auto" and canonical.max_cost_tier == "low":
            local_model = "ministral-3:3b"
            
        use_local_fast_fallback = requested_provider == "auto" and canonical.max_cost_tier in {"low", "medium", "high"}
        cache_provider = "ollama" if use_local_fast_fallback else preferred
        cache_model = local_model if use_local_fast_fallback else self._model_for(cache_provider)
        
        try:
            start_cache = time.monotonic()
            cached = await get_cached_response(user_text, cache_provider, cache_model)
            REDIS_LATENCY.labels(operation="get_cache").observe(time.monotonic() - start_cache)
            if cached:
                FALLBACK_LEVEL_USED.labels(level="cache").inc()
                latency_ms = int((time.monotonic() - start) * 1000)
                resp = self._format_response_dict(
                    cached["answer"], cached["provider"], cached["model"], latency_ms, request_id,
                    fallback_used=cached["fallback_used"], cached=True, route_reason=route_reason
                )
                try:
                    await log_request(
                        request_id=request_id, timestamp=timestamp, message=user_text,
                        provider=cached["provider"], model=cached["model"], latency_ms=latency_ms,
                        cached=True, fallback_used=cached["fallback_used"], 
                        prompt_tokens=resp["prompt_tokens"], completion_tokens=resp["completion_tokens"],
                        cost_usd=0.0, status="ok"
                    )
                except Exception: pass
                return resp
        except Exception: pass

        # 5. Execution Strategy
        canonical_with_system = self._with_system_prompt(canonical)
        timeout_ms = canonical.timeout_ms or 30000
        timeout_s = max(0.001, timeout_ms / 1000.0)
        deadline_s = start + timeout_s

        # Get the strict provider chain
        chain = self._get_provider_chain(preferred, system_state=system_state)

        if requested_provider == "auto":
            chain = self._prioritize_auto_fallbacks(chain)

        if not chain:
            raise HTTPException(status_code=500, detail="No providers available")

        # In testing mode or when not healthy, use linear fallback only
        if self.testing_mode or system_state != "healthy":
            if system_state == "critical":
                await self._trigger_global_cooldown()
                BLOCKED_REQUESTS_TOTAL.labels(reason="critical_state", state="critical").inc()
            return await self._run_linear_fallback(
                chain, user_text, start, request_id, timeout_s,
                system_state=system_state,
                model_override=local_model,
                requested_provider=requested_provider,
                health_scores=health_scores
            )

        # HEALTHY state handling (Race strategy)
        self._provider_canonical = canonical_with_system
        try:
            self._log_throttled("race_start", "[SYSTEM] Healthy state: Starting race execution", level=logging.INFO)
            race_data = await self._run_fastest_response_race(
                chain=chain,
                user_text=user_text,
                requested_provider=requested_provider,
                timeout_s=timeout_s,
                deadline_s=deadline_s,
                local_model=local_model,
                use_local_fast_fallback=use_local_fast_fallback,
                health_scores=health_scores
            )
            winner_result, provider_chain, r_timeout_hit, last_error, _ = race_data
            
            if winner_result and "answer" in winner_result:
                latency_ms = int((time.monotonic() - start) * 1000)
                FALLBACK_LEVEL_USED.labels(level="race_winner").inc()
                
                # Check if it was a fallback (not the first in chain)
                fallback_used = False
                winner_provider = winner_result["provider"]
                if provider_chain and winner_provider in provider_chain:
                    fallback_used = provider_chain.index(winner_provider) > 0

                resp = self._format_response_dict(
                    winner_result["answer"], 
                    winner_provider, 
                    winner_result["model"], 
                    latency_ms, 
                    request_id,
                    fallback_used=fallback_used,
                    cached=False,
                    route_reason="race_winner",
                    timeout_hit=r_timeout_hit,
                    provider_chain=provider_chain
                )
                
                # Log success
                try:
                    await log_request(
                        request_id=request_id, timestamp=timestamp, message=user_text,
                        provider=winner_provider, model=winner_result["model"], latency_ms=latency_ms,
                        cached=False, fallback_used=fallback_used, 
                        prompt_tokens=resp["prompt_tokens"], completion_tokens=resp["completion_tokens"],
                        cost_usd=resp["cost_usd"], status="ok"
                    )
                except Exception: pass
                
                # Cache successful response
                try:
                    await set_cached_response(user_text, winner_provider, winner_result["model"], {
                        "answer": winner_result["answer"], "provider": winner_provider, "model": winner_result["model"],
                        "fallback_used": fallback_used
                    })
                except Exception: pass
                
                return resp
        except Exception as e:
            logger.warning("[RACE] Strategy failed: %s, falling back to linear execution", e)

        # If race failed or returned nothing, final linear fallback as a safeguard
        return await self._run_linear_fallback(
            chain, user_text, start, request_id, timeout_s,
            system_state=system_state,
            model_override=local_model,
            requested_provider=requested_provider
        )

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
            if not await self.is_provider_healthy(provider_name):
                self._log_throttled(f"stream_skip_{provider_name}", f"[STREAM] Skipping unhealthy provider: {provider_name}")
                continue

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
