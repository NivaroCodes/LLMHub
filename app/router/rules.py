from __future__ import annotations

from dataclasses import dataclass


LOCAL_KEYWORDS = ("цена", "наличие", "адрес", "да", "нет")
GEMINI_KEYWORDS = ("сравни", "проанализируй", "жалоба")


@dataclass(frozen=True)
class RouteDecision:
    provider: str
    model: str
    route_reason: str


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def choose_route(
    prompt: str,
    *,
    local_model: str = "local_llama3_8b",
    gemini_model: str = "gemini-1.5-flash",
    default_model: str = "gpt-4o-mini",
) -> RouteDecision:
    normalized = _normalize(prompt)
    is_short = len(normalized) < 100
    has_local_keyword = any(keyword in normalized for keyword in LOCAL_KEYWORDS)
    has_gemini_keyword = any(keyword in normalized for keyword in GEMINI_KEYWORDS)

    if is_short and has_local_keyword:
        return RouteDecision(
            provider="ollama",
            model=local_model,
            route_reason="short_prompt_with_commerce_keyword",
        )
    if has_gemini_keyword:
        return RouteDecision(
            provider="gemini",
            model=gemini_model,
            route_reason="analysis_or_complaint_keyword",
        )
    return RouteDecision(
        provider="openai",
        model=default_model,
        route_reason="default_fallback",
    )
