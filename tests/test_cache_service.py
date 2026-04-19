from app.services.cache_service import build_cache_key


class TestBuildCacheKey:
    def test_deterministic(self):
        k1 = build_cache_key("hello", "gemini", "gemini-2.0-flash")
        k2 = build_cache_key("hello", "gemini", "gemini-2.0-flash")
        assert k1 == k2

    def test_normalizes_case(self):
        k1 = build_cache_key("HELLO", "gemini", "gemini-2.0-flash")
        k2 = build_cache_key("hello", "gemini", "gemini-2.0-flash")
        assert k1 == k2

    def test_normalizes_whitespace(self):
        k1 = build_cache_key("  hello  ", "gemini", "gemini-2.0-flash")
        k2 = build_cache_key("hello", "gemini", "gemini-2.0-flash")
        assert k1 == k2

    def test_different_messages_produce_different_keys(self):
        k1 = build_cache_key("hello", "gemini", "gemini-2.0-flash")
        k2 = build_cache_key("world", "gemini", "gemini-2.0-flash")
        assert k1 != k2

    def test_different_providers_produce_different_keys(self):
        k1 = build_cache_key("hello", "gemini", "model")
        k2 = build_cache_key("hello", "ollama", "model")
        assert k1 != k2

    def test_different_models_produce_different_keys(self):
        k1 = build_cache_key("hello", "gemini", "gemini-2.0-flash")
        k2 = build_cache_key("hello", "gemini", "gemini-1.5-flash")
        assert k1 != k2

    def test_key_has_correct_prefix(self):
        key = build_cache_key("hello", "gemini", "model")
        assert key.startswith("chat_cache:")

    def test_key_is_string(self):
        key = build_cache_key("hello", "gemini", "model")
        assert isinstance(key, str)
