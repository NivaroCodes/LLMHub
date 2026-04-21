class TestRuleRoute:
    def test_short_prompt_with_commerce_keyword_routes_to_ollama(self, llm_service):
        assert llm_service._rule_route("какая цена?") == "ollama"

    def test_analysis_keyword_routes_to_gemini(self, llm_service):
        assert llm_service._rule_route("сравни два варианта") == "gemini"

    def test_default_routes_to_openai(self, llm_service):
        assert llm_service._rule_route("Hello, how are you?") == "openai"


class TestProviderChain:
    def test_openai_preference_preserves_full_fallback_order(self, llm_service):
        chain = llm_service._get_provider_chain("openai")

        assert [name for name, _ in chain] == ["openai", "openrouter", "gemini", "ollama"]

    def test_openrouter_preference_preserves_its_fallback_order(self, llm_service):
        chain = llm_service._get_provider_chain("openrouter")

        assert [name for name, _ in chain] == ["openai", "openrouter", "gemini", "ollama"]

    def test_ollama_preference_keeps_local_first_then_gemini(self, llm_service):
        chain = llm_service._get_provider_chain("ollama")

        assert [name for name, _ in chain] == ["ollama", "gemini", "openai", "openrouter"]

    def test_auto_preference_uses_default_fallback_order(self, llm_service):
        chain = llm_service._get_provider_chain("auto")

        assert [name for name, _ in chain] == ["ollama", "gemini", "openai", "openrouter"]

    def test_chain_skips_providers_without_credentials(self, llm_service_factory):
        service = llm_service_factory(
            gemini_api_key=None,
            openai_api_key=None,
            openrouter_api_key=None,
        )

        chain = service._get_provider_chain("openai")

        assert [name for name, _ in chain] == ["ollama"]

    def test_model_for_known_providers(self, llm_service):
        assert llm_service._model_for("gemini") == llm_service.gemini_model
        assert llm_service._model_for("openai") == llm_service.openai_model
        assert llm_service._model_for("openrouter") == llm_service.openrouter_model
        assert llm_service._model_for("ollama") == llm_service.ollama_model

    def test_model_for_unknown_provider_returns_unknown(self, llm_service):
        assert llm_service._model_for("nonexistent") == "unknown"
