import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.llm_services import LLMService
from app.core.canonical import CanonicalChatRequest

@pytest.fixture
def llm_service():
    with patch('app.services.llm_services.OpenAIProvider', return_value=MagicMock()), \
         patch('app.services.llm_services.GeminiProvider', return_value=MagicMock()), \
         patch('app.services.llm_services.OpenRouterProvider', return_value=MagicMock()), \
         patch('app.services.llm_services.OllamaProvider', return_value=MagicMock()):
        # Mock env vars
        with patch.dict('os.environ', {
            "OPENAI_API_KEY": "sk-test",
            "GEMINI_API_KEY": "test",
            "OPENROUTER_API_KEY": "test"
        }):
            service = LLMService()
            return service

@pytest.mark.asyncio
async def test_state_machine_locked(llm_service):
    # Mock redis to return cooldown_active
    mock_redis = AsyncMock()
    now = time.time()
    # MGET order: 0: cooldown, 1: openai_state, 2: openai_quota, 3: openai_p95...
    vals = [None] * 13
    vals[0] = str(now + 100)
    mock_redis.mget.return_value = vals
    
    with patch.object(llm_service, '_get_redis_safe', return_value=mock_redis):
        status = await llm_service._get_system_state()
        assert status["state"] == "locked"
        assert status["cooldown"] is True

@pytest.mark.asyncio
async def test_fallback_chain_strict_order_degraded(llm_service):
    # In degraded state, chain should be strictly OpenAI -> Gemini -> OpenRouter -> Ollama
    chain = llm_service._get_provider_chain("ollama", system_state="degraded")
    names = [name for name, _ in chain]
    # Check that all expected providers are there in order
    assert names == ["openai", "gemini", "openrouter", "ollama"]

@pytest.mark.asyncio
async def test_guaranteed_static_response_when_all_failed(llm_service):
    # Mock all providers to fail
    llm_service._call_provider = AsyncMock(side_effect=Exception("API Down"))
    llm_service._get_system_state = AsyncMock(return_value={
        "state": "critical", "cooldown": False, "scores": {}
    })
    
    # Use text that definitely doesn't match static intent
    payload = CanonicalChatRequest.from_text("qwertyuiop123456")
    response = await llm_service.get_response(payload)
    
    assert response["provider"] == "static_fallback"
    assert "Извините" in response["answer"]
    assert response["fallback_used"] is True

@pytest.mark.asyncio
async def test_race_disabled_in_critical(llm_service):
    # Mock system state as critical
    llm_service._get_system_state = AsyncMock(return_value={
        "state": "critical", "cooldown": False, "scores": {"ollama": 0}
    })
    
    with patch.object(llm_service, '_run_fastest_response_race', AsyncMock()) as mock_race, \
         patch.object(llm_service, '_run_linear_fallback', AsyncMock(return_value={"answer": "ok", "provider": "p"})) as mock_linear:
        
        await llm_service.get_response("test")
        
        mock_race.assert_not_called()
        mock_linear.assert_called()

@pytest.mark.asyncio
async def test_global_rate_limiting_dependency():
    from app.dependencies import rate_limiter
    from fastapi import HTTPException
    
    mock_request = MagicMock()
    mock_redis = AsyncMock()
    mock_request.app.state.redis = mock_redis
    mock_request.client.host = "1.2.3.4"
    
    # Simulate IP limit exceeded
    mock_redis.eval.return_value = [61, 10]
    with pytest.raises(HTTPException) as exc:
        await rate_limiter(mock_request)
    assert exc.value.status_code == 429
    assert "IP rate limit exceeded" in exc.value.detail
    
    # Simulate Global limit exceeded
    mock_redis.eval.return_value = [10, 301]
    with pytest.raises(HTTPException) as exc:
        await rate_limiter(mock_request)
    assert exc.value.status_code == 429
    assert "Global rate limit exceeded" in exc.value.detail
