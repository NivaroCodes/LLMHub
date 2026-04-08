import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

_WINDOW_SECONDS = 60
_MAX_REQUESTS = 60
_clients: dict[str, deque[float]] = defaultdict(deque)


async def rate_limiter(request: Request) -> dict:
    now = time.monotonic()
    key = request.client.host if request.client else "unknown"
    bucket = _clients[key]

    while bucket and now - bucket[0] > _WINDOW_SECONDS:
        bucket.popleft()

    if len(bucket) >= _MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    bucket.append(now)
    return {"remaining": _MAX_REQUESTS - len(bucket)}
