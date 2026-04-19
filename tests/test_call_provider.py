from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_call_provider_passes_timeout_only_to_ollama(llm_service_factory):
    service = llm_service_factory()
    service._model_for = MagicMock(return_value="llama3:8b")
    provider = MagicMock()
    provider.get_completion = AsyncMock(return_value="reply from ollama")

    result = await service._call_provider(
        provider_name="ollama",
        provider=provider,
        user_text="hello",
        timeout_s=30,
    )

    service._model_for.assert_called_once_with("ollama")
    provider.get_completion.assert_awaited_once_with("hello", model="llama3:8b", timeout_s=30)
    assert result == "reply from ollama"


@pytest.mark.asyncio
async def test_call_provider_omits_timeout_for_remote_providers(llm_service_factory):
    service = llm_service_factory()
    service._model_for = MagicMock(return_value="gpt-4o-mini")
    provider = MagicMock()
    provider.get_completion = AsyncMock(return_value="reply from openai")

    result = await service._call_provider(
        provider_name="openai",
        provider=provider,
        user_text="hi",
        timeout_s=99,
    )

    service._model_for.assert_called_once_with("openai")
    provider.get_completion.assert_awaited_once_with("hi", model="gpt-4o-mini")
    assert result == "reply from openai"
