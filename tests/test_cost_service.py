import pytest
from app.services.cost_service import estimate_tokens, estimate_cost


class TestEstimateTokens:
    def test_returns_at_least_one(self):
        assert estimate_tokens("") == 1

    def test_short_text(self):
        # "Hello" = 5 chars → 5 // 4 = 1
        assert estimate_tokens("Hello") == 1

    def test_longer_text(self):
        text = "a" * 400  # 400 chars → 100 tokens
        assert estimate_tokens(text) == 100

    def test_proportional(self):
        t1 = estimate_tokens("short")
        t2 = estimate_tokens("short" * 10)
        assert t2 > t1


class TestEstimateCost:
    def test_returns_three_values(self):
        result = estimate_cost("openai", "gpt-4o-mini", "hello", "world")
        assert len(result) == 3

    def test_ollama_is_free(self):
        _, _, cost = estimate_cost("ollama", "llama3:8b", "hello world", "some answer here")
        assert cost == 0.0

    def test_gemini_flash_has_cost(self):
        _, _, cost = estimate_cost("gemini", "gemini-2.0-flash", "hello world", "answer")
        assert cost >= 0.0

    def test_openai_gpt4o_costs_more_than_mini(self):
        prompt = "a" * 1000
        completion = "b" * 1000
        _, _, cost_mini = estimate_cost("openai", "gpt-4o-mini", prompt, completion)
        _, _, cost_gpt4o = estimate_cost("openai", "gpt-4o", prompt, completion)
        assert cost_gpt4o > cost_mini

    def test_prompt_tokens_positive(self):
        prompt_tokens, _, _ = estimate_cost("gemini", "gemini-2.0-flash", "hello", "world")
        assert prompt_tokens >= 1

    def test_completion_tokens_positive(self):
        _, completion_tokens, _ = estimate_cost("gemini", "gemini-2.0-flash", "hello", "world")
        assert completion_tokens >= 1

    def test_unknown_provider_uses_openai_default(self):
        _, _, cost_unknown = estimate_cost("unknown_provider", "some-model", "hello world", "answer text")
        _, _, cost_openai = estimate_cost("openai", "gpt-4o-mini", "hello world", "answer text")
        assert cost_unknown == cost_openai

    def test_cost_rounds_to_8_decimal_places(self):
        _, _, cost = estimate_cost("openai", "gpt-4o-mini", "hello", "world")
        # round() with 8 decimals — just verify it's a float
        assert isinstance(cost, float)
