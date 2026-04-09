import os
from typing import Optional

import redis.asyncio as redis

_redis_client: Optional[redis.Redis] = None


async def init_redis() -> redis.Redis:
    global _redis_client

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _redis_client = redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    await _redis_client.ping()
    return _redis_client


async def close_redis() -> None:
    global _redis_client

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


def get_redis() -> redis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis client is not initialized")
    return _redis_client
