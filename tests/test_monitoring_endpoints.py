from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import monitoring


def test_monitoring_overview_returns_payload():
    app = FastAPI()
    app.include_router(monitoring.router)
    client = TestClient(app)

    payload = {
        "window_minutes": 60,
        "kpis": {"total_requests": 42},
        "providers": [{"provider": "ollama", "requests": 20, "errors": 1, "avg_latency_ms": 150, "cost_usd": 0.0}],
    }
    with patch.object(monitoring, "get_monitoring_overview", AsyncMock(return_value=payload)):
        response = client.get("/monitoring/overview?window_minutes=60")

    assert response.status_code == 200
    assert response.json() == payload


def test_monitoring_timeseries_returns_points():
    app = FastAPI()
    app.include_router(monitoring.router)
    client = TestClient(app)

    series = [{"bucket_start": "2026-04-24T12:00:00+00:00", "requests": 3, "errors": 0, "avg_latency_ms": 90, "p95_latency_ms": 140}]
    with patch.object(monitoring, "get_monitoring_timeseries", AsyncMock(return_value=series)):
        response = client.get("/monitoring/timeseries?window_minutes=180&bucket_minutes=5")

    assert response.status_code == 200
    assert response.json()["points"] == series


def test_monitoring_failures_returns_items():
    app = FastAPI()
    app.include_router(monitoring.router)
    client = TestClient(app)

    failures = [{"timestamp": "2026-04-24T12:03:00+00:00", "provider": "openai", "model": "gpt-4o-mini", "status": "error", "error": "timeout", "latency_ms": 1000, "request_id": "r1"}]
    with patch.object(monitoring, "get_recent_failures", AsyncMock(return_value=failures)):
        response = client.get("/monitoring/failures?limit=20")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["items"] == failures
