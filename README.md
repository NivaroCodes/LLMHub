# LLMHub

LLMHub is a lightweight gateway for routing chat requests between providers.
It exposes a simple HTTP API and a CLI so you can use the same core for local
experiments, developer workflows, and end-user apps.

## Features

- FastAPI backend with a single `/chat` endpoint.
- Provider routing: Gemini and Ollama (OpenAI optional).
- Rule-based or agent-based routing for `preferred_provider="auto"`.
- CLI for quick local usage.
- `.env` configuration with safe defaults.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in keys.
4. Run the server:

```bash
uvicorn app.main:app --reload
```

Open the docs at `http://127.0.0.1:8000/docs`.

## Configuration

Environment variables (see `.env.example`):

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (default: `gemini-3-flash-preview`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3:8b`)
- `ROUTER_MODE` (`rules` or `agent`)
- `ROUTER_MODEL` (model used by the router in agent mode)

Note: `.env` is ignored by git and must stay private.

## API

### `POST /chat`

Request body:

```json
{
  "message": "Hello",
  "preferred_provider": "auto",
  "max_cost_tier": "low",
  "timeout_ms": 120000
}
```

Response body:

```json
{
  "answer": "string",
  "provider": "gemini|ollama",
  "model": "string",
  "latency_ms": 0,
  "request_id": "uuid",
  "fallback_used": false
}
```

## CLI

The CLI is provided via `llmhub` after an editable install.

Install:

```bash
pip install -e .
```

Commands:

```bash
llmhub chat "Hello" --provider auto
llmhub chat "Hello" --provider gemini --json
llmhub serve --host 0.0.0.0 --port 8000 --reload
```

## Routing Modes

`ROUTER_MODE=rules`:
Simple heuristics based on message length and keywords.

`ROUTER_MODE=agent`:
Gemini is used as a router model to decide `gemini` vs `ollama`.
This adds latency and consumes tokens for routing.

## Troubleshooting

- If Ollama replies with 404, the model is not downloaded:
  `ollama pull llama3:8b`
- If Gemini returns 404, the model name is not available for your project.
  Use the Gemini API `models.list` to see what is enabled for your key.

