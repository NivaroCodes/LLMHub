from __future__ import annotations
import json
import time
import uuid
from typing import List
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from app.core.canonical import CanonicalChatRequest
from app.schemas.openai_chat import (
    OpenAIChatChoice,
    OpenAIChatChoiceMessage,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIUsage,
)
from app.services.llm_services import get_llm_service

router = APIRouter(prefix="/v1", tags=["openai-compatible"])
# Shared process-wide singleton (see app.services.llm_services.get_llm_service).
# Same instance as `app.api.endpoints.llm_service`.
llm_service = get_llm_service()


@router.post("/chat/completions", response_model=OpenAIChatCompletionResponse)
async def create_chat_completion(payload: OpenAIChatCompletionRequest, request: Request):
    canonical = CanonicalChatRequest.from_openai_chat(payload)
    tenant_type = request.headers.get("x-business-type")
    tenant_name = request.headers.get("x-business-name", "")
    metadata = {}
    if tenant_type:
        metadata = {"business_type": tenant_type, "business_name": tenant_name}
    canonical = canonical.model_copy(update={"metadata": metadata})
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if canonical.stream:
        async def event_stream():
            async for chunk in llm_service.stream_response(canonical):
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": chunk["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk["content"]},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = await llm_service.get_response(canonical)
    content = result["answer"]
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]

    return OpenAIChatCompletionResponse(
        id=chat_id,
        created=created,
        model=result["model"],
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatChoiceMessage(content=content),
                finish_reason="stop",
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@router.get("/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    models = [
        {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
        {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
        {"id": "gpt-4-turbo", "object": "model", "owned_by": "openai"},
        {"id": "gemini-2.0-flash", "object": "model", "owned_by": "google"},
        {"id": "gemini-1.5-flash", "object": "model", "owned_by": "google"},
        {"id": "llama3:8b", "object": "model", "owned_by": "ollama"},
        {"id": "llama3:70b", "object": "model", "owned_by": "ollama"},
        {"id": "mistral:7b", "object": "model", "owned_by": "ollama"},
    ]
    
    return {
        "object": "list",
        "data": models
    }
