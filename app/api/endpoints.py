from fastapi import APIRouter, Depends

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
