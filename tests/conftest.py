from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def llm_service():
    with (
        patch("app.services.llm_services.GeminiProvider"),
        patch("app.services.llm_services.OpenAIProvider"),
        patch("app.services.llm_services.OpenRouterProvider"),
        patch("app.services.llm_services.OllamaProvider"),
        patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY":     "fake-gemini",
                "OPENAI_API_KEY":     "fake-openai",
                "OPENROUTER_API_KEY": "fake-openrouter",
                "ROUTER_MODE":        "rules",
            },
        ),
    ):
        from app.services.llm_services import LLMService
        yield LLMService()
