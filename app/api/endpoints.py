from fastapi import APIRouter, Depends
from app.dependencies import rate_limiter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_services import LLMService


router = APIRouter()
llm_service = LLMService()


@router.post("/chat")
async def chat(payload: ChatRequest, response_model=ChatResponse, limit = Depends(rate_limiter)):
	user_text = payload.message
	response = await llm_service.get_response(user_text)
	return response


	