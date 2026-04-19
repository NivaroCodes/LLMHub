from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest

from app.services import llm_services as llm_module


@pytest.fixture
def llm_service_factory(monkeypatch):
    def build(
        *,
        gemini_api_key: str | None = "fake-gemini",
        openai_api_key: str | None = "fake-openai",
        openrouter_api_key: str | None = "fake-openrouter",
        router_mode: str = "rules",
        router_model: str = "router-test-model",
        gemini_provider=None,
        openai_provider=None,
        openrouter_provider=None,
        ollama_provider=None,
    ):
        env_values = {
            "GEMINI_API_KEY": gemini_api_key,
            "OPENAI_API_KEY": openai_api_key,
            "OPENROUTER_API_KEY": openrouter_api_key,
            "OPENAI_MODEL": "gpt-4o-mini",
            "GEMINI_MODEL": "gemini-2.0-flash",
            "OPENROUTER_MODEL": "openai/gpt-4o-mini",
            "OLLAMA_MODEL": "llama3:8b",
            "OLLAMA_BASE_URL": "http://ollama.test",
            "ROUTER_MODE": router_mode,
            "ROUTER_MODEL": router_model,
        }

        for key, value in env_values.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

        gemini_provider = gemini_provider or AsyncMock(name="gemini_provider")
        openai_provider = openai_provider or AsyncMock(name="openai_provider")
        openrouter_provider = openrouter_provider or AsyncMock(name="openrouter_provider")
        ollama_provider = ollama_provider or AsyncMock(name="ollama_provider")

        with ExitStack() as stack:
            gemini_cls = stack.enter_context(patch.object(llm_module, "GeminiProvider"))
            openai_cls = stack.enter_context(patch.object(llm_module, "OpenAIProvider"))
            openrouter_cls = stack.enter_context(patch.object(llm_module, "OpenRouterProvider"))
            ollama_cls = stack.enter_context(patch.object(llm_module, "OllamaProvider"))

            gemini_cls.return_value = gemini_provider
            openai_cls.return_value = openai_provider
            openrouter_cls.return_value = openrouter_provider
            ollama_cls.return_value = ollama_provider

            service = llm_module.LLMService()

        return service

    return build


@pytest.fixture
def llm_service(llm_service_factory):
    return llm_service_factory()
