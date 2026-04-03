import os
import uuid
from datetime import datetime

from app.providers.openai_client import OpenAIProvider
from app.providers.ollama_client import OllamaProvider

class LLMService():
	def __init__(self):
		self.openai_api_key = os.getenv("OPENAI_API_KEY")
		self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
		self.openai = OpenAIProvider(api_key=self.openai_api_key) if self.openai_api_key else None
		self.ollama = OllamaProvider(base_url=self.ollama_base_url)

	async def get_response(self, user_text: str) -> dict:
		if (len(user_text) > 100 or "write a code" in user_text.lower()) and self.openai is not None:
			selected_model = "gpt-4o-mini"
			answer = await self.openai.get_completion(user_text, model=selected_model)
		else:
			selected_model = "llama3"
			answer = await self.ollama.get_completion(user_text, model=selected_model)

		return {
			"user_said": user_text,
			"model_used": selected_model,
			"provider": answer,
			"model": selected_model,
			"latency_ms": datetime.now().strftime("%H:%M:%S.%f"),
			"request_id": uuid.uuid4(),
			"fallback_used": False,
			"actual_answer": answer
		}
