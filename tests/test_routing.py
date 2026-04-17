class TestRuleRoute:
    def test_short_generic_goes_to_ollama(self, llm_service):
        assert llm_service._rule_route("Hello, how are you?") == "ollama"

    def test_long_text_goes_to_gemini(self, llm_service):
        assert llm_service._rule_route("x" * 161) == "gemini"

    def test_exactly_160_chars_goes_to_ollama(self, llm_service):
        assert llm_service._rule_route("a" * 160) == "ollama"

    def test_keyword_code_en(self, llm_service):
        assert llm_service._rule_route("write some code for me") == "gemini"

    def test_keyword_code_ru(self, llm_service):
        assert llm_service._rule_route("напиши код на Python") == "gemini"

    def test_keyword_explain_ru(self, llm_service):
        assert llm_service._rule_route("объясни что такое REST API") == "gemini"

    def test_keyword_algorithm_ru(self, llm_service):
        assert llm_service._rule_route("алгоритм сортировки пузырьком") == "gemini"

    def test_keyword_plan_ru(self, llm_service):
        assert llm_service._rule_route("составь план разработки") == "gemini"

    def test_keyword_compare_ru(self, llm_service):
        assert llm_service._rule_route("сравни PostgreSQL и MongoDB") == "gemini"

    def test_case_insensitive(self, llm_service):
        assert llm_service._rule_route("Write CODE for me") == "gemini"


class TestProviderChain:
    def test_gemini_preferred_leads_chain(self, llm_service):
        chain = llm_service._get_provider_chain("gemini")
        assert chain[0][0] == "gemini"

    def test_ollama_preferred_leads_chain(self, llm_service):
        chain = llm_service._get_provider_chain("ollama")
        assert chain[0][0] == "ollama"

    def test_openai_preferred_leads_chain(self, llm_service):
        chain = llm_service._get_provider_chain("openai")
        assert chain[0][0] == "openai"

    def test_openrouter_preferred_leads_chain(self, llm_service):
        chain = llm_service._get_provider_chain("openrouter")
        assert chain[0][0] == "openrouter"

    def test_chain_has_no_none_providers(self, llm_service):
        for preferred in ("gemini", "ollama", "openai", "openrouter", "auto"):
            chain = llm_service._get_provider_chain(preferred)
            for name, provider in chain:
                assert provider is not None, f"None provider in chain for preferred={preferred}"

    def test_chain_not_empty(self, llm_service):
        chain = llm_service._get_provider_chain("auto")
        assert len(chain) > 0

    def test_model_for_known_providers(self, llm_service):
        assert llm_service._model_for("gemini")     == llm_service.gemini_model
        assert llm_service._model_for("openai")     == llm_service.openai_model
        assert llm_service._model_for("openrouter") == llm_service.openrouter_model
        assert llm_service._model_for("ollama")     == llm_service.ollama_model

    def test_model_for_unknown_returns_unknown(self, llm_service):
        assert llm_service._model_for("nonexistent") == "unknown"
