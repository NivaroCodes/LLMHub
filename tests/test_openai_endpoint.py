from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from app.api import openai_endpoints


def test_openai_compatible_endpoint_returns_openai_like_json():
    app = FastAPI()
    app.include_router(openai_endpoints.router)
    openai_endpoints.llm_service.get_response = AsyncMock(
        return_value={
            "answer": "hello from provider",
            "provider": "ollama",
            "model": "llama3:8b",
            "prompt_tokens": 2,
            "completion_tokens": 4,
            "latency_ms": 120,
            "request_id": "test-id",
            "fallback_used": False,
            "cached": False,
            "cost_usd": 0.0,
            "route_reason": "short_commerce",
        }
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "llama3:8b"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in body
