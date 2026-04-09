from fastapi import HTTPException, Request

_WINDOW_SECONDS = 60
_MAX_REQUESTS = 60


async def rate_limiter(request: Request) -> dict:
    redis_client = request.app.state.redis
    client_host = request.client.host if request.client else "unknown"
    key = f"rate_limit:{client_host}"

    current_count = await redis_client.incr(key)
    if current_count == 1:
        await redis_client.expire(key, _WINDOW_SECONDS)

    if current_count > _MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return {"remaining": _MAX_REQUESTS - current_count}
