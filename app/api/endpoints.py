import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request

from app.db.database import get_stats
from app.dependencies import rate_limiter
from app.metrics import metrics_response, record_chat_metrics
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_services import LLMService

router = APIRouter()
llm_service = LLMService()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    limit: dict = Depends(rate_limiter),
):
    try:
        response = await llm_service.get_response(payload)
    except HTTPException as exc:
        record_chat_metrics(
            provider="unknown",
            status="error",
            cached=False,
            fallback_used=False,
            latency_ms=0,
        )
        raise exc

    record_chat_metrics(
        provider=response["provider"],
        status="ok",
        cached=response["cached"],
        fallback_used=response["fallback_used"],
        latency_ms=response["latency_ms"],
    )
    return response

APP_START_TIME = datetime.now(timezone.utc)

@router.get("/metrics")
async def metrics():
    return metrics_response()


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - start_time

        method = request.method
        endpoint = request.scope.get("route")

        REQUESTS_TOTAL.labels(method, endpoint).inc()

        if status_code >= 400:
            ERRORS_TOTAL.labels(method, endpoint, str(status_code)).inc()

        LATENCY.labels(method, endpoint).observe(duration)

    return response

@router.get("/health")
async def health_checker(request: Request):
    current_time = datetime.now(timezone.utc)
    uptime = current_time - APP_START_TIME

    redis_client = request.app.state.redis
    try:
        await redis_client.ping()
        status = "ok"
        redis_status = "available"
        reason = None
    except Exception as e:
        status = "degraded"
        redis_status = "unavailable"
        reason = str(e)

    health_status = {
        "status": status,
        "redis": redis_status,
        "timestamp": current_time.isoformat(),
        "uptime": str(uptime).split(".")[0],
        "start_time": APP_START_TIME.isoformat(),
    }
    if reason is not None:
        health_status["reason"] = reason

    return health_status


@router.get("/stats")
async def stats():
    return await get_stats()
