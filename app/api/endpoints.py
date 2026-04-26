import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.canonical import CanonicalChatRequest
from app.db.database import get_stats
from app.dependencies import rate_limiter
from app.metrics import metrics_response, record_chat_metrics, LATENCY_BREAKDOWN
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_services import LLMService

router = APIRouter()
llm_service = LLMService()


def legacy_to_canonical(payload: ChatRequest) -> CanonicalChatRequest:
    return CanonicalChatRequest.from_legacy_chat(payload)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    limit: dict = Depends(rate_limiter),
):
    business_type = request.headers.get("x-business-type")
    business_name = request.headers.get("x-business-name", "")
    metadata = {}
    if business_type:
        metadata = {"business_type": business_type, "business_name": business_name}
    canonical = legacy_to_canonical(payload).model_copy(update={"metadata": metadata})
    try:
        start_total = time.monotonic()
        response = await llm_service.get_response(canonical)
        LATENCY_BREAKDOWN.labels(stage="endpoint_total").observe(time.monotonic() - start_total)
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
