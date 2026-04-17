from fastapi import APIRouter, Depends, Request
from datetime import datetime, timezone
from app.db.database import get_stats
from app.dependencies import rate_limiter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_services import LLMService

router = APIRouter()
llm_service = LLMService()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    limit: dict = Depends(rate_limiter),
):
    response = await llm_service.get_response(payload)
    return response

APP_START_TIME = datetime.now(timezone.utc)


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
