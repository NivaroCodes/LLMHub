from app.core.canonical import CanonicalChatRequest
from app.schemas.chat import ChatRequest
from app.schemas.openai_chat import OpenAIChatCompletionRequest, OpenAIChatMessage


def test_legacy_and_openai_payloads_produce_same_canonical_object():
    legacy = ChatRequest(message="какая цена?", preferred_provider="auto", timeout_ms=120000)
    openai = OpenAIChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[OpenAIChatMessage(role="user", content="какая цена?")],
        timeout_ms=120000,
    )

    canonical_legacy = CanonicalChatRequest.from_legacy_chat(legacy)
    canonical_openai = CanonicalChatRequest.from_openai_chat(openai).model_copy(
        update={"preferred_provider": "auto", "max_cost_tier": "low"}
    )

    assert canonical_legacy == canonical_openai
