from fastapi import APIRouter, Depends
from datetime import datetime, timezone
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
async def health_checker():
    current_time = datetime.now(timezone.utc)
    uptime = current_time - APP_START_TIME

    return {
        "status": "ok",
        "timestamp": current_time.isoformat(),
        "uptime": str(uptime).split(".")[0],
        "start_time": APP_START_TIME.isoformat(),
    }


