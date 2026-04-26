from fastapi import HTTPException, Request

_WINDOW_SECONDS = 60
_MAX_REQUESTS_PER_IP = 60
_MAX_REQUESTS_GLOBAL = 300


async def rate_limiter(request: Request) -> dict:
    from app.metrics import REDIS_LATENCY, REQUEST_BLOCKED_BY_RATE_LIMIT
    import time
    
    redis_client = request.app.state.redis
    client_host = request.client.host if request.client else "unknown"
    ip_key = f"rate_limit:ip:{client_host}"
    global_key = "rate_limit:global"

    start = time.monotonic()
    # Using Lua script to check both IP and Global limits in one go
    lua_script = """
    local ip_count = redis.call('INCR', KEYS[1])
    if ip_count == 1 then
        redis.call('EXPIRE', KEYS[1], ARGV[1])
    end
    
    local global_count = redis.call('INCR', KEYS[2])
    if global_count == 1 then
        redis.call('EXPIRE', KEYS[2], ARGV[1])
    end
    
    return {ip_count, global_count}
    """
    counts = await redis_client.eval(lua_script, 2, ip_key, global_key, _WINDOW_SECONDS)
    ip_count, global_count = counts
    REDIS_LATENCY.labels(operation="rate_limit_lua").observe(time.monotonic() - start)

    if ip_count > _MAX_REQUESTS_PER_IP:
        REQUEST_BLOCKED_BY_RATE_LIMIT.labels(provider="api_gateway_ip").inc()
        raise HTTPException(status_code=429, detail="IP rate limit exceeded")

    if global_count > _MAX_REQUESTS_GLOBAL:
        REQUEST_BLOCKED_BY_RATE_LIMIT.labels(provider="api_gateway_global").inc()
        raise HTTPException(status_code=429, detail="Global rate limit exceeded")

    return {"ip_count": ip_count, "global_count": global_count}
