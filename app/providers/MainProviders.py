from abc import ABC, abstractmethod
from typing import AsyncIterator


class MainProvider(ABC):
    @abstractmethod
    async def get_completion(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    async def get_streaming_completion(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError


BaseProvider = MainProvider
