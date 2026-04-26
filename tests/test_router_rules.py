from app.router.rules import RouteDecision, choose_route


def test_short_prompt_with_price_keyword_routes_to_local():
    decision = choose_route("Какая цена на услугу?")
    assert decision.provider == "ollama"
    assert decision.model == "local_llama3_8b"
    assert decision.route_reason


def test_short_prompt_with_availability_keyword_routes_to_local():
    decision = choose_route("Есть наличие сегодня?")
    assert decision.provider == "ollama"
    assert decision.route_reason


def test_short_prompt_with_address_keyword_routes_to_local():
    decision = choose_route("Какой адрес филиала?")
    assert decision.provider == "ollama"
    assert decision.route_reason


def test_short_prompt_with_yes_keyword_routes_to_local():
    decision = choose_route("Да, можно сегодня?")
    assert decision.provider == "ollama"
    assert decision.route_reason


def test_short_prompt_with_no_keyword_routes_to_local():
    decision = choose_route("Нет, тогда завтра?")
    assert decision.provider == "ollama"
    assert decision.route_reason


def test_compare_keyword_routes_to_gemini():
    decision = choose_route("Сравни два предложения по стоимости")
    assert decision.provider == "gemini"
    assert decision.model == "gemini-1.5-flash"
    assert decision.route_reason


def test_analyze_keyword_routes_to_gemini():
    decision = choose_route("Проанализируй это сообщение клиента")
    assert decision.provider == "gemini"
    assert decision.route_reason


def test_complaint_keyword_routes_to_gemini():
    decision = choose_route("У нас жалоба от клиента по заказу")
    assert decision.provider == "gemini"
    assert decision.route_reason


def test_default_routes_to_openrouter_when_no_keywords_match():
    decision = choose_route("Привет, расскажи про вашу компанию")
    assert decision.provider == "openrouter"
    assert decision.model == "openai/gpt-4o-mini"
    assert decision.route_reason


def test_local_rule_has_higher_priority_than_gemini_keywords_for_short_prompt():
    decision = choose_route("Цена и сравни варианты")
    assert decision == RouteDecision(
        provider="ollama",
        model="local_llama3_8b",
        route_reason="short_prompt_with_commerce_keyword",
    )
