from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.schemas.chat import ChatRequest
from app.services import llm_services as llm_module


@pytest.mark.asyncio
async def test_get_response_returns_cached_payload_without_calling_provider(llm_service_factory):
    service = llm_service_factory()
    cached_payload = {
        "answer": "cached answer",
        "provider": "gemini",
        "model": service.gemini_model,
        "fallback_used": False,
    }
    service._call_provider = AsyncMock()

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=cached_payload)) as get_cached,
        patch.object(llm_module, "set_cached_response", AsyncMock()) as set_cached,
        patch.object(llm_module, "log_request", AsyncMock()) as log_request,
    ):
        response = await service.get_response(
            ChatRequest(message="write code", preferred_provider="gemini")
        )

    get_cached.assert_awaited_once_with("write code", "gemini", service.gemini_model)
    service._call_provider.assert_not_awaited()
    set_cached.assert_not_awaited()
    assert response["answer"] == "cached answer"
    assert response["provider"] == "gemini"
    assert response["model"] == service.gemini_model
    assert response["cached"] is True
    assert response["cost_usd"] == 0.0
    assert log_request.await_args.kwargs["cached"] is True


@pytest.mark.asyncio
async def test_get_response_calls_preferred_provider_and_caches_result(llm_service_factory):
    service = llm_service_factory()
    service._call_provider = AsyncMock(return_value="fresh answer")

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()) as set_cached,
        patch.object(llm_module, "log_request", AsyncMock()) as log_request,
    ):
        response = await service.get_response(
            ChatRequest(message="write code", preferred_provider="gemini", timeout_ms=45000)
        )

    service._call_provider.assert_awaited_once()
    assert service._call_provider.await_args.args[0] == "gemini"
    assert service._call_provider.await_args.args[1] == service.gemini
    assert service._call_provider.await_args.args[2] == "write code"
    assert service._call_provider.await_args.args[3] == pytest.approx(10.0, abs=0.1)
    assert response["answer"] == "fresh answer"
    assert response["provider"] == "gemini"
    assert response["model"] == service.gemini_model
    assert response["fallback_used"] is False
    assert response["cached"] is False
    set_cached.assert_awaited_once_with(
        "write code",
        "gemini",
        service.gemini_model,
        {
            "answer": "fresh answer",
            "provider": "gemini",
            "model": service.gemini_model,
            "fallback_used": False,
        },
    )
    assert log_request.await_args.kwargs["status"] == "ok"


@pytest.mark.asyncio
async def test_get_response_accepts_string_payload_and_uses_rule_router_for_auto(llm_service_factory):
    service = llm_service_factory(router_mode="rules")
    service._rule_route = AsyncMock()
    service._rule_route = lambda user_text: "openai"
    service._call_provider = AsyncMock(return_value="local answer")

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()),
        patch.object(llm_module, "log_request", AsyncMock()),
    ):
        response = await service.get_response("hello")

    service._call_provider.assert_awaited_once_with(
        "ollama", service.ollama, "hello", 3.0, model_override="ministral-3:3b"
    )
    assert response["provider"] == "ollama"
    assert response["model"] == "ministral-3:3b"
    assert response["answer"] == "local answer"


@pytest.mark.asyncio
async def test_get_response_uses_agent_router_when_auto_mode_is_enabled(llm_service_factory):
    service = llm_service_factory(router_mode="agent")
    service._agent_route = AsyncMock(return_value="gemini")
    service._call_provider = AsyncMock(return_value="agent answer")

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()),
        patch.object(llm_module, "log_request", AsyncMock()),
    ):
        response = await service.get_response(ChatRequest(message="route me", preferred_provider="auto"))

    service._agent_route.assert_awaited_once_with("route me")
    service._call_provider.assert_awaited_once_with(
        "ollama", service.ollama, "route me", 3.0, model_override="ministral-3:3b"
    )
    assert response["provider"] == "ollama"


@pytest.mark.asyncio
async def test_get_response_auto_low_uses_openrouter_after_local_timeout_even_if_agent_chose_gemini(llm_service_factory):
    service = llm_service_factory(router_mode="agent")
    service._agent_route = AsyncMock(return_value="gemini")

    async def call_provider(provider_name, provider, user_text, timeout_s, **kwargs):
        if provider_name == "ollama":
            raise TimeoutError("Provider timeout 35000ms")
        return "fallback answer"

    service._call_provider = AsyncMock(side_effect=call_provider)

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()),
        patch.object(llm_module, "log_request", AsyncMock()),
    ):
        response = await service.get_response(
            ChatRequest(message="Привет. Расскажи про Нью-Йорк", preferred_provider="auto", max_cost_tier="low")
        )

    assert [call.args[0] for call in service._call_provider.await_args_list] == ["ollama", "openrouter"]
    assert response["provider"] == "openrouter"
    assert response["fallback_used"] is True


@pytest.mark.asyncio
async def test_get_response_races_primary_provider_then_falls_back(llm_service_factory):
    service = llm_service_factory()

    async def call_provider(provider_name, provider, user_text, timeout_s, **kwargs):
        if provider_name == "gemini":
            raise RuntimeError("gemini unavailable")
        return "fallback answer"

    service._call_provider = AsyncMock(side_effect=call_provider)

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()) as set_cached,
        patch.object(llm_module, "log_request", AsyncMock()) as log_request,
    ):
        response = await service.get_response(
            ChatRequest(message="write code", preferred_provider="gemini", max_cost_tier="none", timeout_ms=1200)
        )

    assert [call.args[0] for call in service._call_provider.await_args_list] == ["gemini", "openrouter"]
    assert response["answer"] == "fallback answer"
    assert response["provider"] == "openrouter"
    assert response["fallback_used"] is True
    assert set_cached.await_args.args[1] == "openrouter"
    assert log_request.await_args.kwargs["fallback_used"] is True


@pytest.mark.asyncio
async def test_get_response_ignores_cache_read_failures_and_continues(llm_service_factory):
    service = llm_service_factory(openrouter_api_key=None)
    service._call_provider = AsyncMock(return_value="fresh answer")

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(side_effect=RuntimeError("redis down"))),
        patch.object(llm_module, "set_cached_response", AsyncMock()),
        patch.object(llm_module, "log_request", AsyncMock()),
    ):
        response = await service.get_response(ChatRequest(message="write code", preferred_provider="gemini"))

    assert response["answer"] == "fresh answer"
    assert response["cached"] is False
    assert response["provider"] == "gemini"


@pytest.mark.asyncio
async def test_get_response_raises_502_after_all_providers_fail(llm_service_factory):
    service = llm_service_factory()
    service._call_provider = AsyncMock(side_effect=RuntimeError("provider down"))

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()) as set_cached,
        patch.object(llm_module, "log_request", AsyncMock()) as log_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await service.get_response(
                ChatRequest(message="write code", preferred_provider="gemini")
            )

    assert exc_info.value.status_code == 502
    assert "provider down" in exc_info.value.detail
    assert service._call_provider.await_count == 4
    set_cached.assert_not_awaited()
    assert log_request.await_args.kwargs["status"] == "error"
    assert log_request.await_args.kwargs["provider"] == "ollama"


@pytest.mark.asyncio
async def test_get_response_raises_502_for_non_404_http_errors(llm_service_factory):
    service = llm_service_factory()
    request = httpx.Request("POST", "http://provider.test/chat")
    response = httpx.Response(500, request=request)
    error = httpx.HTTPStatusError("boom", request=request, response=response)
    service._call_provider = AsyncMock(side_effect=error)

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()) as set_cached,
        patch.object(llm_module, "log_request", AsyncMock()) as log_request,
        patch.object(llm_module.asyncio, "sleep", AsyncMock()),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await service.get_response(ChatRequest(message="hello", preferred_provider="openrouter"))

    assert exc_info.value.status_code == 502
    assert "boom" in exc_info.value.detail
    set_cached.assert_not_awaited()
    assert log_request.await_args.kwargs["status"] == "error"


@pytest.mark.asyncio
async def test_get_response_surfaces_missing_ollama_model_as_400(llm_service_factory):
    service = llm_service_factory()
    request = httpx.Request("POST", "http://ollama.test/api/generate")
    response = httpx.Response(404, request=request)
    error = httpx.HTTPStatusError("missing model", request=request, response=response)
    service._call_provider = AsyncMock(side_effect=error)

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()) as set_cached,
        patch.object(llm_module, "log_request", AsyncMock()) as log_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await service.get_response(ChatRequest(message="hello", preferred_provider="ollama"))

    assert exc_info.value.status_code == 400
    assert "ollama pull llama3:8b" in exc_info.value.detail
    set_cached.assert_not_awaited()
    log_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_response_ignores_cache_write_failures(llm_service_factory):
    service = llm_service_factory()
    service._call_provider = AsyncMock(return_value="fresh answer")

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock(side_effect=RuntimeError("cache write failed"))),
        patch.object(llm_module, "log_request", AsyncMock()) as log_request,
    ):
        response = await service.get_response(ChatRequest(message="hello", preferred_provider="gemini"))

    assert response["answer"] == "fresh answer"
    assert log_request.await_args.kwargs["status"] == "ok"


@pytest.mark.asyncio
async def test_get_response_ignores_success_log_failures(llm_service_factory):
    service = llm_service_factory()
    service._call_provider = AsyncMock(return_value="fresh answer")

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()),
        patch.object(llm_module, "log_request", AsyncMock(side_effect=RuntimeError("db down"))),
    ):
        response = await service.get_response(ChatRequest(message="hello", preferred_provider="gemini"))

    assert response["answer"] == "fresh answer"
    assert response["provider"] == "gemini"


@pytest.mark.asyncio
async def test_get_response_ignores_cache_hit_log_failures(llm_service_factory):
    service = llm_service_factory()
    cached_payload = {
        "answer": "cached answer",
        "provider": "gemini",
        "model": service.gemini_model,
        "fallback_used": False,
    }

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=cached_payload)),
        patch.object(llm_module, "set_cached_response", AsyncMock()),
        patch.object(llm_module, "log_request", AsyncMock(side_effect=RuntimeError("db down"))),
    ):
        response = await service.get_response(ChatRequest(message="hello", preferred_provider="gemini"))

    assert response["answer"] == "cached answer"
    assert response["cached"] is True


@pytest.mark.asyncio
async def test_get_response_ignores_error_log_failures_before_raising_502(llm_service_factory):
    service = llm_service_factory()
    service._call_provider = AsyncMock(side_effect=RuntimeError("provider down"))

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()),
        patch.object(llm_module, "log_request", AsyncMock(side_effect=RuntimeError("db down"))),
        patch.object(llm_module.asyncio, "sleep", AsyncMock()),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await service.get_response(ChatRequest(message="hello", preferred_provider="gemini"))

    assert exc_info.value.status_code == 502
    assert "provider down" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_response_raises_500_when_no_providers_are_available(llm_service_factory):
    service = llm_service_factory(gemini_api_key=None, openai_api_key=None, openrouter_api_key=None)
    service.ollama = None

    with pytest.raises(HTTPException) as exc_info:
        await service.get_response(ChatRequest(message="hello", preferred_provider="ollama"))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "No providers available"


@pytest.mark.asyncio
async def test_get_response_disables_provider_on_binary_policy_error(llm_service_factory):
    service = llm_service_factory(openrouter_api_key=None)
    calls: list[str] = []

    async def call_provider(provider_name, provider, user_text, timeout_s, **kwargs):
        calls.append(provider_name)
        if provider_name == "openai":
            raise RuntimeError("DLL load failed while importing jiter: policy blocked")
        return "ok from fallback"

    service._call_provider = AsyncMock(side_effect=call_provider)

    with (
        patch.object(llm_module, "get_cached_response", AsyncMock(return_value=None)),
        patch.object(llm_module, "set_cached_response", AsyncMock()),
        patch.object(llm_module, "log_request", AsyncMock()),
    ):
        first = await service.get_response(
            ChatRequest(message="hello", preferred_provider="openai", max_cost_tier="none", timeout_ms=30000)
        )
        second = await service.get_response(
            ChatRequest(message="hello again", preferred_provider="openai", max_cost_tier="none", timeout_ms=30000)
        )

    assert first["provider"] == "gemini"
    assert second["provider"] == "gemini"
    assert "openai" in service._disabled_providers
    assert calls[0] == "openai"
