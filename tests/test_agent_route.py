from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_agent_route_uses_rule_router_when_gemini_is_unavailable(llm_service_factory):
    service = llm_service_factory(gemini_api_key=None)
    service._rule_route = MagicMock(return_value="ollama")

    result = await service._agent_route("Hi")

    service._rule_route.assert_called_once_with("Hi")
    assert result == "ollama"


@pytest.mark.asyncio
async def test_agent_route_asks_router_model_and_normalizes_gemini_response(llm_service_factory):
    gemini_provider = MagicMock()
    gemini_provider.get_completion = AsyncMock(return_value="  GEMINI \n")
    service = llm_service_factory(
        router_mode="agent",
        router_model="router-test-model",
        gemini_provider=gemini_provider,
    )

    result = await service._agent_route("Write a sorting algorithm")

    gemini_provider.get_completion.assert_awaited_once()
    prompt = gemini_provider.get_completion.await_args.args[0]
    assert "Return ONLY one word: gemini or ollama." in prompt
    assert "Write a sorting algorithm" in prompt
    assert gemini_provider.get_completion.await_args.kwargs == {"model": "router-test-model"}
    assert result == "gemini"


@pytest.mark.asyncio
async def test_agent_route_returns_ollama_when_router_selects_it(llm_service_factory):
    gemini_provider = MagicMock()
    gemini_provider.get_completion = AsyncMock(return_value=" ollama ")
    service = llm_service_factory(gemini_provider=gemini_provider)

    result = await service._agent_route("Say hello")

    assert result == "ollama"


@pytest.mark.asyncio
async def test_agent_route_falls_back_to_rules_when_router_errors(llm_service_factory):
    gemini_provider = MagicMock()
    gemini_provider.get_completion = AsyncMock(side_effect=RuntimeError("router timeout"))
    service = llm_service_factory(gemini_provider=gemini_provider)
    service._rule_route = MagicMock(return_value="ollama")

    result = await service._agent_route("Hi")

    service._rule_route.assert_called_once_with("Hi")
    assert result == "ollama"


@pytest.mark.asyncio
async def test_agent_route_falls_back_to_rules_when_router_returns_unknown_label(llm_service_factory):
    gemini_provider = MagicMock()
    gemini_provider.get_completion = AsyncMock(return_value="openai")
    service = llm_service_factory(gemini_provider=gemini_provider)
    service._rule_route = MagicMock(return_value="ollama")

    result = await service._agent_route("Hi")

    service._rule_route.assert_called_once_with("Hi")
    assert result == "ollama"
