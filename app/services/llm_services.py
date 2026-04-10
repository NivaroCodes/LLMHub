import os
import time
import uuid

import httpx
from fastapi import HTTPException

from app.providers.gemini_client import GeminiProvider
from app.providers.openai_client import OpenAIProvider
from app.providers.ollama_client import OllamaProvider
from app.schemas.chat import ChatRequest
from app.services.cache_service import get_cached_response, set_cached_response

class LLMService:
	def __init__(self):
		self.openai_api_key = os.getenv("OPENAI_API_KEY")
		self.gemini_api_key = os.getenv("GEMINI_API_KEY")
		self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
		self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
		self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
		self.router_mode = os.getenv("ROUTER_MODE", "rules").strip().lower()
		self.router_model = os.getenv("ROUTER_MODEL", self.gemini_model)
		self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3:8b")
		self.openai = OpenAIProvider(api_key=self.openai_api_key) if self.openai_api_key else None
		self.gemini = GeminiProvider(api_key=self.gemini_api_key, default_model=self.gemini_model) if self.gemini_api_key else None
		self.ollama = OllamaProvider(base_url=self.ollama_base_url, default_model=self.ollama_model)

	def _rule_route(self, user_text: str) -> str:
		text = user_text.lower()
		if len(user_text) > 160:
			return "gemini"
		if any(token in text for token in ("код", "code", "алгоритм", "объясни", "сравни", "план")):
			return "gemini"
		return "ollama"

	async def _agent_route(self, user_text: str) -> str:
		if self.gemini is None:
			return self._rule_route(user_text)
		prompt = (
			"You are a routing agent for an LLM gateway. "
			"Choose the best provider for the user request. "
			"Return ONLY one word: gemini or ollama.\n\n"
			f"User request:\n{user_text}"
		)
		try:
			decision = await self.gemini.get_completion(prompt, model=self.router_model)
		except Exception:
			return self._rule_route(user_text)
		value = decision.strip().lower()
		if "ollama" in value:
			return "ollama"
		if "gemini" in value:
			return "gemini"
		return self._rule_route(user_text)

	def _select_provider(self, preferred: str) -> tuple[str, object, bool]:
		fallback_used = False
		if preferred == "openai":
			if self.gemini is not None:
				return "gemini", self.gemini, True
			if self.ollama is not None:
				return "ollama", self.ollama, True
			raise HTTPException(status_code=500, detail="No providers available")
		if preferred == "gemini":
			if self.gemini is not None:
				return "gemini", self.gemini, False
			if self.ollama is not None:
				return "ollama", self.ollama, True
			raise HTTPException(status_code=500, detail="No providers available")
		if preferred == "ollama":
			return "ollama", self.ollama, False
		return "ollama", self.ollama, True

	async def get_response(self, payload: ChatRequest | str) -> dict:
		start = time.monotonic()
		if isinstance(payload, ChatRequest):
			user_text = payload.message
			preferred = (payload.preferred_provider or "auto").strip().lower()
			timeout_s = max(1, int(payload.timeout_ms / 1000))
		else:
			user_text = str(payload)
			preferred = "auto"
			timeout_s = 30

		if preferred == "auto":
			if self.router_mode == "agent":
				preferred = await self._agent_route(user_text)
			else:
				preferred = self._rule_route(user_text)

		provider_name, provider, fallback_used = self._select_provider(preferred)

		if provider_name == "gemini":
			model_used = self.gemini_model
		else:
			model_used = self.ollama_model

		try:
			cached_response = await get_cached_response(user_text, provider_name, model_used)
		except Exception:
			cached_response = None

		if cached_response is not None:
			latency_ms = int((time.monotonic() - start) * 1000)
			return {
				"answer": cached_response["answer"],
				"provider": cached_response["provider"],
				"model": cached_response["model"],
				"latency_ms": latency_ms,
				"request_id": str(uuid.uuid4()),
				"fallback_used": cached_response["fallback_used"],
				"cached": True,
			}

		if provider_name == "gemini":
			answer = await provider.get_completion(user_text, model=model_used)
		else:
			try:
				answer = await provider.get_completion(user_text, model=model_used, timeout_s=timeout_s)
			except httpx.HTTPStatusError as exc:
				if exc.response.status_code == 404:
					raise HTTPException(
						status_code=400,
						detail=f"Ollama model '{model_used}' not found. Run: ollama pull {model_used}",
					) from exc
				raise

		latency_ms = int((time.monotonic() - start) * 1000)

		response_payload = {
			"answer": answer,
			"provider": provider_name,
			"model": model_used,
			"fallback_used": fallback_used,
		}
		try:
			await set_cached_response(user_text, provider_name, model_used, response_payload)
		except Exception:
			pass

		return {
			"answer": answer,
			"provider": provider_name,
			"model": model_used,
			"latency_ms": latency_ms,
			"request_id": str(uuid.uuid4()),
			"fallback_used": fallback_used,
			"cached": False,
		}
