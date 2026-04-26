import time
import asyncio
from fastapi import Request
from app.metrics import EVENT_LOOP_LAG, LATENCY_BREAKDOWN, record_chat_metrics
from starlette.middleware.base import BaseHTTPMiddleware

class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ["/metrics", "/health", "/static"]:
            return await call_next(request)
        
        start_time = time.monotonic()
        
        # Event Loop Lag detection
        loop = asyncio.get_event_loop()
        start_loop = loop.time()
        await asyncio.sleep(0)
        lag = loop.time() - start_loop
        EVENT_LOOP_LAG.set(lag)
        
        # Track request start
        try:
            response = await call_next(request)
            
            process_time = time.monotonic() - start_time
            LATENCY_BREAKDOWN.labels(stage="total_http").observe(process_time)
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            process_time = time.monotonic() - start_time
            LATENCY_BREAKDOWN.labels(stage="total_http_error").observe(process_time)
            raise e
