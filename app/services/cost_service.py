_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "gemini": {
        "gemini-2.0-flash":          (0.075,  0.30),
        "gemini-2.0-flash-lite":     (0.075,  0.30),
        "gemini-1.5-flash":          (0.075,  0.30),
        "gemini-3-flash-preview":    (0.075,  0.30),
        "default":                   (0.075,  0.30),
    },
    "openai": {
        "gpt-4o-mini":               (0.15,   0.60),
        "gpt-4o":                    (2.50,  10.00),
        "gpt-4-turbo":               (10.00, 30.00),
        "default":                   (0.15,   0.60),
    },
    "openrouter": {
        "openai/gpt-4o-mini":        (0.15,   0.60),
        "openai/gpt-4o":             (2.50,  10.00),
        "anthropic/claude-3-haiku":  (0.25,   1.25),
        "default":                   (0.50,   1.50),
    },
    "ollama": {
        "default":                   (0.0,    0.0),
    },
}


def estimate_tokens(text: str) -> int:
    """Approximate token count: ~4 characters per token."""
    return max(1, len(text) // 4)


def estimate_cost(
    provider: str,
    model: str,
    prompt: str,
    completion: str,
) -> tuple[int, int, float]:
    """
    Returns (prompt_tokens, completion_tokens, cost_usd).
    Cost is 0.0 for local providers (ollama).
    """
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(completion)

    provider_table = _PRICING.get(provider, _PRICING["openai"])
    input_price, output_price = provider_table.get(model, provider_table["default"])

    cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    return prompt_tokens, completion_tokens, round(cost, 8)