# LLMHub Integration Guide

This guide covers all ways to integrate with LLMHub: Python SDK, OpenAI-compatible API, and HTTP API.

## 1. Python SDK

### Installation

```bash
pip install llmhub
```

Or install from source:

```bash
git clone https://github.com/NivaroCodes/LLMHub.git
cd LLMHub
pip install -e .
```

### Quick Start

```python
from llmhub import LLMHub

hub = LLMHub()
response = hub.chat_sync("Hello from Kazakhstan!")
print(response.answer)
```

### With Configuration

```python
from llmhub import LLMHub

config = {
    "providers": ["ollama", "gemini", "openrouter"],
    "fallback_strategy": "cost_optimized",
    "max_cost_usd": 0.01
}

hub = LLMHub(config=config)
response = hub.chat_sync("Tell me a story", config=config)
print(f"Provider: {response.provider}")
print(f"Latency: {response.latency_ms}ms")
print(f"Answer: {response.answer}")
```

### Streaming

```python
import asyncio
from llmhub import LLMHub

async def stream_example():
    hub = LLMHub()
    async for chunk in hub.stream_chat("Tell me a story"):
        print(chunk, end="", flush=True)
    await hub.close()

asyncio.run(stream_example())
```

### Async Usage

```python
import asyncio
from llmhub import LLMHub

async def async_example():
    hub = LLMHub()
    response = await hub.chat("Hello!")
    print(response.answer)
    await hub.close()

asyncio.run(async_example())
```

### Context Manager

```python
from llmhub import LLMHub

with LLMHub() as hub:
    response = hub.chat_sync("Hello!")
    print(response.answer)
```

### Custom Base URL

```python
from llmhub import LLMHub

# For local development
hub = LLMHub(base_url="http://localhost:8000")

# For custom deployment
hub = LLMHub(base_url="https://your-domain.com")
```

### Response Object

```python
from llmhub import LLMHub

hub = LLMHub()
response = hub.chat_sync("Hello!")

print(f"Answer: {response.answer}")
print(f"Provider: {response.provider}")
print(f"Model: {response.model}")
print(f"Latency: {response.latency_ms}ms")
print(f"Cost: ${response.cost_usd}")
print(f"Fallback used: {response.fallback_used}")
print(f"Cached: {response.cached}")
print(f"Request ID: {response.request_id}")
```

### Convenience Function

```python
from llmhub import create_hub

hub = create_hub(
    base_url="https://llmhub-production.up.railway.app",
    config={"providers": ["ollama", "gemini"]}
)
response = hub.chat_sync("Hello!")
```

## 2. OpenAI-Compatible API

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://llmhub-production.up.railway.app/v1",
    api_key="any-key"  # LLMHub doesn't validate API keys
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Streaming with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://llmhub-production.up.railway.app/v1",
    api_key="any-key"
)

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Using LangChain

```python
from langchain.llms import OpenAI

llm = OpenAI(
    base_url="https://llmhub-production.up.railway.app/v1",
    api_key="any"
)

result = llm("What is machine learning?")
print(result)
```

### Using LangChain with Chat Models

```python
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    base_url="https://llmhub-production.up.railway.app/v1",
    api_key="any"
)

response = chat.predict("What is machine learning?")
print(response)
```

### List Models

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://llmhub-production.up.railway.app/v1",
    api_key="any-key"
)

models = client.models.list()
for model in models.data:
    print(f"{model.id} ({model.owned_by})")
```

### curl Examples

#### Chat Completion

```bash
curl -X POST https://llmhub-production.up.railway.app/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer any-key' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### Streaming

```bash
curl -X POST https://llmhub-production.up.railway.app/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

#### List Models

```bash
curl https://llmhub-production.up.railway.app/v1/models
```

## 3. HTTP API

### Using curl

```bash
curl -X POST https://llmhub-production.up.railway.app/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Hello!",
    "preferred_provider": "auto"
  }'
```

### With Configuration

```bash
curl -X POST https://llmhub-production.up.railway.app/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Tell me a story",
    "preferred_provider": "ollama",
    "max_cost_tier": "low"
  }'
```

### Streaming

```bash
curl -X POST https://llmhub-production.up.railway.app/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Tell me a story",
    "stream": true
  }'
```

### JavaScript / TypeScript

```javascript
const response = await fetch('https://llmhub-production.up.railway.app/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Hello!',
    preferred_provider: 'auto'
  })
})

const data = await response.json()
console.log(data.answer)
```

### Streaming in JavaScript

```javascript
const response = await fetch('https://llmhub-production.up.railway.app/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Tell me a story',
    stream: true
  })
})

const reader = response.body.getReader()
const decoder = new TextDecoder()

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  const chunk = decoder.decode(value)
  console.log(chunk)
}
```

### Python with httpx

```python
import httpx

client = httpx.Client()
response = client.post(
    "https://llmhub-production.up.railway.app/chat",
    json={
        "message": "Hello!",
        "preferred_provider": "auto"
    }
)

data = response.json()
print(data["answer"])
```

### Streaming in Python with httpx

```python
import httpx

client = httpx.Client()
with client.stream(
    "POST",
    "https://llmhub-production.up.railway.app/chat",
    json={"message": "Tell me a story", "stream": true}
) as response:
    for line in response.iter_lines():
        if line:
            print(line.decode())
```

## API Reference

### HTTP API Endpoints

#### POST /chat

Send a chat request to LLMHub.

**Request:**
```json
{
  "message": "Hello!",
  "preferred_provider": "auto",
  "max_cost_tier": "low",
  "stream": false
}
```

**Response:**
```json
{
  "answer": "Hello! How can I help?",
  "provider": "gemini",
  "model": "gemini-2.0-flash",
  "latency_ms": 120,
  "cost_usd": 0.0001,
  "fallback_used": false,
  "cached": false,
  "request_id": "uuid"
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "redis": "available",
  "timestamp": "2026-05-10T10:00:00Z",
  "uptime": "0:05:30"
}
```

#### GET /metrics

Prometheus metrics endpoint.

### OpenAI-Compatible Endpoints

#### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint.

**Request:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4o-mini",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

#### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "gpt-4o", "object": "model", "owned_by": "openai"},
    {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
    {"id": "gemini-2.0-flash", "object": "model", "owned_by": "google"},
    {"id": "llama3:8b", "object": "model", "owned_by": "ollama"}
  ]
}
```

## Configuration Options

### Provider Selection

- `preferred_provider`: "auto" (default), "ollama", "gemini", "openrouter", "openai"
- `max_cost_tier`: "low", "medium", "high"

### Routing

- `ROUTER_MODE`: "rules" (default), "agent"
- `ROUTER_MODEL`: Model for agent-based routing

### Timeouts

- `LOCAL_TIMEOUT_MS`: Timeout for local providers (default: 3000)
- `REMOTE_TIMEOUT_MS`: Timeout for remote providers (default: 10000)
- `ROUTER_TIMEOUT_MS`: Timeout for routing decision (default: 2000)

### Caching

- `CACHE_TTL_SECONDS`: Cache time-to-live (default: 300)

## Error Handling

### Python SDK

```python
from llmhub import LLMHub
import httpx

hub = LLMHub()

try:
    response = hub.chat_sync("Hello!")
    print(response.answer)
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
except httpx.RequestError as e:
    print(f"Request error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### OpenAI SDK

```python
from openai import OpenAI
from openai import APIError, APITimeoutError

client = OpenAI(
    base_url="https://llmhub-production.up.railway.app/v1",
    api_key="any-key"
)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
except APIError as e:
    print(f"API error: {e}")
except APITimeoutError as e:
    print(f"Timeout error: {e}")
```

## Examples

### Simple Chat Bot

```python
from llmhub import LLMHub

hub = LLMHub()

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    response = hub.chat_sync(user_input)
    print(f"LLMHub: {response.answer}")
```

### Cost-Optimized Application

```python
from llmhub import LLMHub

config = {
    "preferred_provider": "auto",
    "max_cost_tier": "low"
}

hub = LLMHub(config=config)
response = hub.chat_sync("What is the cheapest way to travel?")
print(response.answer)
```

### Quality-First Application

```python
from llmhub import LLMHub

config = {
    "preferred_provider": "openai",
    "max_cost_tier": "high"
}

hub = LLMHub(config=config)
response = hub.chat_sync("Write a detailed essay")
print(response.answer)
```

## Support

For integration issues:
- GitHub Issues: https://github.com/NivaroCodes/LLMHub/issues
- Email: admin@yourdomain.com
- Live API: https://llmhub-production.up.railway.app
