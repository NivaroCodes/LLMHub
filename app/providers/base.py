from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.core.canonical import CanonicalChatRequest


class BaseChatProvider(ABC):
    @abstractmethod
    async def chat(self, request: CanonicalChatRequest, model: str | None = None, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self, request: CanonicalChatRequest, model: str | None = None, **kwargs
    ) -> AsyncIterator[str]:
        raise NotImplementedError
