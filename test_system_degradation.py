import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.services.llm_services import LLMService
from app.core.canonical import CanonicalChatRequest

@pytest.fixture
def llm_service():
    with patch('app.services.llm_services.OpenAIProvider'), \
         patch('app.services.llm_services.GeminiProvider'), \
         patch('app.services.llm_services.OpenRouterProvider'), \
         patch('app.services.llm_services.OllamaProvider'):
        service = LLMService()
        service.testing_mode = True
        return service

@pytest.mark.asyncio
async def test_system_degradation_logic(llm_service):
    mock_redis = AsyncMock()
    mock_redis.incrby.return_value = 1
    
    with patch.object(llm_service, '_get_redis_safe', return_value=mock_redis):
        # 1. Healthy
        mock_redis.mget.return_value = [None] * 13
        status = await llm_service._get_system_state()
        assert status["state"] == "healthy"
        assert status["active_count"] == 4

        # 2. Degraded (2 failures: openai=index 1, gemini=index 4)
        vals = [None] * 13
        vals[1] = "1" 
        vals[4] = "1"
        mock_redis.mget.return_value = vals
        
        status = await llm_service._get_system_state()
        assert status["state"] == "degraded"
        assert status["active_count"] == 2

        # 3. Critical (All 4 failed: 1, 4, 7, 10)
        vals = [None] * 13
        for i in [1, 4, 7, 10]: vals[i] = "1"
        mock_redis.mget.return_value = vals
        
        status = await llm_service._get_system_state()
        assert status["state"] == "critical"
        assert status["active_count"] == 0
        
        llm_service._call_provider = AsyncMock(side_effect=Exception("Fail"))
        response = await llm_service.get_response("hi")
        assert response["provider"] == "static_fallback"

@pytest.mark.asyncio
async def test_cooldown_locking(llm_service):
    import time
    mock_redis = AsyncMock()
    now = time.time()
    vals = [None] * 13
    vals[0] = str(now + 100) # cooldown_until
    mock_redis.mget.return_value = vals
    mock_redis.incrby.return_value = 1
    
    with patch.object(llm_service, '_get_redis_safe', return_value=mock_redis):
        response = await llm_service.get_response("any")
        assert response["provider"] == "static_fallback"
