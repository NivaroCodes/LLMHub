from app.services.cost_service import estimate_cost, estimate_tokens


class TestEstimateTokens:
    def test_empty_text_still_counts_as_one_token(self):
        assert estimate_tokens("") == 1

    def test_token_estimate_uses_four_character_chunks(self):
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcdefgh") == 2

    def test_longer_text_produces_more_tokens(self):
        assert estimate_tokens("short") < estimate_tokens("short" * 10)


class TestEstimateCost:
    def test_ollama_requests_are_free(self):
        prompt_tokens, completion_tokens, cost = estimate_cost(
            "ollama",
            "llama3:8b",
            "a" * 40,
            "b" * 80,
        )

        assert prompt_tokens == 10
        assert completion_tokens == 20
        assert cost == 0.0

    def test_openai_uses_model_specific_pricing(self):
        prompt_tokens, completion_tokens, cost = estimate_cost(
            "openai",
            "gpt-4o-mini",
            "a" * 400,
            "b" * 400,
        )

        assert prompt_tokens == 100
        assert completion_tokens == 100
        assert cost == 0.000075

    def test_unknown_provider_falls_back_to_openai_default_pricing(self):
        result = estimate_cost("unknown-provider", "custom-model", "a" * 400, "b" * 400)

        assert result == (100, 100, 0.000075)

    def test_unknown_model_uses_provider_default_pricing(self):
        result = estimate_cost("openrouter", "missing-model", "a" * 400, "b" * 400)

        assert result == (100, 100, 0.0002)
