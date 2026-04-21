from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.chat import ChatRequest
from app.schemas.openai_chat import OpenAIChatCompletionRequest


Role = Literal["system", "user", "assistant", "tool"]


class CanonicalMessage(BaseModel):
    role: Role
    content: str


class CanonicalChatRequest(BaseModel):
    messages: list[CanonicalMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    model: str | None = None
    preferred_provider: str = "auto"
    max_cost_tier: str = "low"
    timeout_ms: int = 120000
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None

    @model_validator(mode="after")
    def _validate_request(self) -> "CanonicalChatRequest":
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        if self.temperature is not None and not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        return self

    @property
    def user_text(self) -> str:
        parts = [msg.content.strip() for msg in self.messages if msg.role == "user" and msg.content.strip()]
        return "\n".join(parts).strip()

    @classmethod
    def from_legacy_chat(cls, payload: ChatRequest) -> "CanonicalChatRequest":
        return cls(
            messages=[CanonicalMessage(role="user", content=payload.message)],
            model="gpt-4o-mini",
            preferred_provider=payload.preferred_provider,
            max_cost_tier=payload.max_cost_tier,
            timeout_ms=payload.timeout_ms,
        )

    @classmethod
    def from_openai_chat(cls, payload: OpenAIChatCompletionRequest) -> "CanonicalChatRequest":
        messages = [CanonicalMessage(role=msg.role, content=msg.content) for msg in payload.messages]
        return cls(
            messages=messages,
            stream=payload.stream,
            model=payload.model,
            timeout_ms=payload.timeout_ms,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            user=payload.user,
        )

    @classmethod
    def from_text(cls, text: str) -> "CanonicalChatRequest":
        return cls(messages=[CanonicalMessage(role="user", content=text)])

    def to_provider_messages(self) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def cache_key_message(self) -> str:
        return self.user_text
