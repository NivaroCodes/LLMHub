from pydantic import BaseModel

class ChatRequest(BaseModel):
	message: str
	preferred_provider: str = "ollama"
	max_cost_tier: str = "low"
	timeout_ms: int = 30000

class ChatResponse(BaseModel):
	answer: str
	provider: str
	model: str
	latency_ms: str
	request_id: str
	fallback_used: bool