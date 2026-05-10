"""
LLMHub Python SDK Client
"""

import asyncio
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass
import httpx
from datetime import datetime


@dataclass
class ChatConfig:
    """Configuration for chat requests"""
    providers: List[str]
    fallback_strategy: str = "cost_optimized"
    timeout_ms: Optional[Dict[str, int]] = None
    max_cost_usd: Optional[float] = None


@dataclass
class ChatResponse:
    """Response from LLMHub"""
    answer: str
    provider: str
    model: str
    latency_ms: int
    cost_usd: float
    fallback_used: bool
    cached: bool
    request_id: str


class LLMHub:
    """LLMHub client for multi-provider LLM gateway"""
    
    def __init__(
        self,
        base_url: str = "https://llmhub-production.up.railway.app",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LLMHub client
        
        Args:
            base_url: LLMHub API base URL
            api_key: Optional API key (not currently used)
            config: Default configuration for requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_config = config or {}
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=120.0
            )
        return self._client
    
    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client"""
        if self._sync_client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=120.0
            )
        return self._sync_client
    
    def _merge_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge request config with default config"""
        merged = self.default_config.copy()
        if config:
            merged.update(config)
        return merged
    
    async def chat(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send message to LLMHub (async)
        
        Args:
            message: The message to send
            config: Optional configuration for this request
            **kwargs: Additional request parameters
            
        Returns:
            ChatResponse with the answer and metadata
        """
        client = self._get_async_client()
        merged_config = self._merge_config(config)
        
        payload = {
            "message": message,
            **merged_config,
            **kwargs
        }
        
        response = await client.post("/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            answer=data.get("answer", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            latency_ms=data.get("latency_ms", 0),
            cost_usd=data.get("cost_usd", 0.0),
            fallback_used=data.get("fallback_used", False),
            cached=data.get("cached", False),
            request_id=data.get("request_id", "")
        )
    
    def chat_sync(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send message to LLMHub (synchronous)
        
        Args:
            message: The message to send
            config: Optional configuration for this request
            **kwargs: Additional request parameters
            
        Returns:
            ChatResponse with the answer and metadata
        """
        client = self._get_sync_client()
        merged_config = self._merge_config(config)
        
        payload = {
            "message": message,
            **merged_config,
            **kwargs
        }
        
        response = client.post("/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            answer=data.get("answer", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            latency_ms=data.get("latency_ms", 0),
            cost_usd=data.get("cost_usd", 0.0),
            fallback_used=data.get("fallback_used", False),
            cached=data.get("cached", False),
            request_id=data.get("request_id", "")
        )
    
    async def stream_chat(
        self,
        message: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream chat response (async generator)
        
        Args:
            message: The message to send
            config: Optional configuration for this request
            **kwargs: Additional request parameters
            
        Yields:
            Chunks of the response as they arrive
        """
        client = self._get_async_client()
        merged_config = self._merge_config(config)
        
        payload = {
            "message": message,
            "stream": True,
            **merged_config,
            **kwargs
        }
        
        async with client.stream("POST", "/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    # SSE format: data: {...}
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            content = data.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
    
    async def close(self):
        """Close HTTP clients"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._sync_client:
            self._sync_client.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._sync_client:
            self._sync_client.close()
        return False


def create_hub(
    base_url: str = "https://llmhub-production.up.railway.app",
    api_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> LLMHub:
    """
    Convenience function to create LLMHub client
    
    Args:
        base_url: LLMHub API base URL
        api_key: Optional API key
        config: Default configuration
        
    Returns:
        LLMHub client instance
    """
    return LLMHub(base_url=base_url, api_key=api_key, config=config)
