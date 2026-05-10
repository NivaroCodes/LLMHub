<p align="center">
  <img src="assets/readme-preview.svg" width="700" />
</p>

<h1 align="center">LLMHub</h1>

<p align="center">
  Lightweight gateway for routing LLM requests across providers.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue" />
  <img src="https://img.shields.io/badge/fastapi-backend-green" />
  <img src="https://img.shields.io/badge/license-MIT-black" />
</p>

---

## Overview

**LLMHub** is a lightweight infrastructure layer that routes chat requests
between multiple LLM providers.

It provides a **single API and CLI interface** to:

- switch between providers
- optimize cost and latency
- run locally or in production

Designed for developers building AI-powered apps without vendor lock-in.

---

## Features

- **Single `/chat` endpoint** powered by FastAPI
- **Multi-provider routing** (Gemini, Ollama, optional OpenAI)
- **Auto routing** via rules or LLM-based agent
- **CLI for local workflows**
- **.env-based configuration**
- **Fallback handling & latency tracking**
- **Business-friendly monitoring dashboard** at `/monitoring/dashboard`
- **Operational monitoring APIs** (`/monitoring/overview`,
  `/monitoring/timeseries`, `/monitoring/failures`)

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Fill required keys.

### 3. Run server

```bash
uvicorn app.main:app --reload
```

Docs available at:

```
http://127.0.0.1:8000/docs
```

---

## Example Request

```json
{
	"message": "Hello",
	"preferred_provider": "auto",
	"max_cost_tier": "low",
	"timeout_ms": 120000
}
```

## Example Response

```json
{
	"answer": "string",
	"provider": "gemini",
	"model": "string",
	"latency_ms": 120,
	"request_id": "uuid",
	"fallback_used": false
}
```

---

## CLI

Install:

```bash
pip install -e .
```

Usage:

```bash
llmhub chat "Hello" --provider auto
llmhub serve --reload
```

---

## Routing Modes

### `rules`

- Fast
- Deterministic
- Based on heuristics

### `agent`

- Uses LLM (Gemini) for routing decisions
- More flexible, but adds latency and cost

---

## Configuration

Environment variables:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `ROUTER_MODE`
- `ROUTER_MODEL`

> `.env` must not be committed

---

## Troubleshooting

- **Ollama 404**

  ```bash
  ollama pull llama3:8b
  ```

- **Gemini 404**
  - Check available models via API

---

## Monitoring

Run service and open:

```
http://127.0.0.1:8000/monitoring/dashboard
```

Key endpoints:

- `/metrics` - Prometheus scrape endpoint
- `/monitoring/overview` - KPI snapshot for selected time window
- `/monitoring/timeseries` - request/error/latency series for charting
- `/monitoring/failures` - latest failed requests list

---

## Production Deployment

### Quick Start with Docker

Get LLMHub running in production in 5 minutes on a VPS:

```bash
# Clone the repository
git clone https://github.com/your-org/llmhub.git
cd llmhub

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start all services (includes Caddy reverse proxy with HTTPS)
docker compose -f docker-compose.prod.yml up -d

# Check health
curl http://localhost:8000/health
```

The stack includes:

- **LLMHub API** (port 8000, internal only)
- **Caddy** (reverse proxy, automatic HTTPS, ports 80/443)
- **PostgreSQL** (request logs, monitoring data)
- **Redis** (caching, provider health state)
- **Prometheus** (metrics scraping)

### Full VPS Deployment Guide

For production deployment on Ubuntu 24.04 VPS with domain, HTTPS, firewall,
backups:

**[📖 VPS Deployment Guide](docs/VPS_DEPLOYMENT.md)**

Includes:

- Server setup and Docker installation
- Caddy reverse proxy with automatic HTTPS
- UFW firewall configuration
- Fail2ban SSH protection
- Automated backups
- Log rotation
- Resource limits
- Health checks

### Additional Documentation

- **[Domain Setup Guide](docs/DOMAIN_SETUP.md)** - Configure custom domain with
  Let's Encrypt
- **[Ollama Production Setup](docs/OLLAMA_PRODUCTION.md)** - Local LLM provider
  on Linux
- **[Backup & Restore Guide](docs/BACKUP_RESTORE.md)** - Backup procedures and
  disaster recovery
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Migrate from
  LiteLLM/OpenRouter/OpenAI
- **[Benchmark Results](docs/BENCHMARKS.md)** - Performance benchmarks and
  LiteLLM comparison

### Environment Variables

Required variables:

```bash
# Cloud providers
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash

OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini

# Local provider (Ollama must run separately)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3:8b

# Routing
ROUTER_MODE=rules  # or 'agent'
ROUTER_MODEL=gemini-2.0-flash
ROUTER_TIMEOUT_MS=2000

# Provider timeouts (milliseconds)
LOCAL_TIMEOUT_MS=3000
REMOTE_TIMEOUT_MS=10000
PROVIDER_STAGGER_MS=250

# Redis
REDIS_URL=redis://redis:6379/0
CACHE_TTL_SECONDS=300

# PostgreSQL
DATABASE_URL=postgresql://llmhub:llmhub@postgres:5432/llmhub
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
	"status": "ok",
	"redis": "available",
	"timestamp": "2026-05-07T11:00:00Z",
	"uptime": "0:05:30"
}
```

### Metrics

Prometheus metrics are exposed at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

Access Prometheus UI at http://localhost:9090

Key metrics:

- `llmhub_chat_requests_total` - Total requests per provider
- `llmhub_chat_request_latency_seconds` - Request latency
- `llmhub_provider_state` - Provider health (1=healthy, 0=degraded, -1=banned)
- `llmhub_fallback_level_total` - Fallback usage by level

### Example Usage

Standard API endpoint:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, LLMHub!",
    "preferred_provider": "auto",
    "max_cost_tier": "low"
  }'
```

Response:

```json
{
	"answer": "Hello! How can I help you today?",
	"provider": "gemini",
	"model": "gemini-2.0-flash",
	"latency_ms": 120,
	"request_id": "uuid",
	"fallback_used": false,
	"cached": false
}
```

### OpenAI-Compatible API

Use with OpenAI SDKs:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Monitoring Endpoints

- `/monitoring/overview` - KPI snapshot for time window
- `/monitoring/timeseries` - Request/error/latency series
- `/monitoring/failures` - Latest failed requests
- `/stats` - Historical statistics

### Stopping the Stack

```bash
docker compose -f docker-compose.prod.yml down
```

To remove volumes (deletes all data):

```bash
docker compose -f docker-compose.prod.yml down -v
```

---

## Benchmarks

LLMHub includes a benchmark suite to measure performance and compare with
alternatives.

### Key Results

- **Cache hit latency**: ~50ms (75% improvement over cache miss)
- **p95 latency**: ~250ms
- **Fallback time**: ~50-100ms
- **Success rate**: 95%+ during provider outages

### LiteLLM Comparison

| Metric               | LLMHub  | LiteLLM | Winner  |
| -------------------- | ------- | ------- | ------- |
| Startup complexity   | 9 steps | 7 steps | LiteLLM |
| Docker startup time  | ~30s    | ~20s    | LiteLLM |
| Base memory usage    | ~512MB  | ~200MB  | LiteLLM |
| Latency overhead     | ~10ms   | ~5ms    | LiteLLM |
| Local-first routing  | Yes     | No      | LLMHub  |
| Self-contained stack | Yes     | No      | LLMHub  |

**Analysis:**

- LiteLLM is faster and lighter (app-only)
- LLMHub is self-contained (includes PostgreSQL, Redis, Caddy, Prometheus)
- LLMHub prioritizes local providers (Ollama)
- Choose LLMHub for local-first, self-contained deployment
- Choose LiteLLM for enterprise features (auth, multi-tenancy)

**[📖 Full Benchmark Results](docs/BENCHMARKS.md)**

---

## Roadmap

- [x] Production deployment with Docker
- [x] CI/CD pipeline
- [ ] Grafana dashboard
- [ ] Rate limiting per tenant
- [ ] Plugin system

---

## License

MIT
