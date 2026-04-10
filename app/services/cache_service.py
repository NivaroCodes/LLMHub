import hashlib
import json
import os
from typing import Any

from app.clients.redis_client import get_redis

_CACHE_PREFIX = "chat_cache"
_CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))


def build_cache_key(message: str, provider: str, model: str) -> str:
    normalized_message = message.strip().lower()
    payload = f"{provider}:{model}:{normalized_message}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{_CACHE_PREFIX}:{digest}"


async def get_cached_response(message: str, provider: str, model: str) -> dict[str, Any] | None:
    redis_client = get_redis()
    key = build_cache_key(message, provider, model)
    cached_value = await redis_client.get(key)
    if cached_value is None:
        return None
    return json.loads(cached_value)


async def set_cached_response(message: str, provider: str, model: str, response: dict[str, Any]) -> None:
    redis_client = get_redis()
    key = build_cache_key(message, provider, model)
    await redis_client.set(key, json.dumps(response, ensure_ascii=False), ex=_CACHE_TTL_SECONDS)

