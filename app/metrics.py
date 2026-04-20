from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response


CHAT_REQUESTS_TOTAL = Counter(
    "llmhub_chat_requests_total",
    "Total number of chat requests handled by the API.",
    labelnames=("provider", "status", "cached", "fallback_used"),
)

CHAT_REQUEST_LATENCY_SECONDS = Histogram(
    "llmhub_chat_request_latency_seconds",
    "End-to-end latency of chat requests.",
    labelnames=("provider", "status"),
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 60, 120),
)


def record_chat_metrics(
    *,
    provider: str,
    status: str,
    cached: bool,
    fallback_used: bool,
    latency_ms: int,
) -> None:
    CHAT_REQUESTS_TOTAL.labels(
        provider=provider,
        status=status,
        cached=str(cached).lower(),
        fallback_used=str(fallback_used).lower(),
    ).inc()
    CHAT_REQUEST_LATENCY_SECONDS.labels(
        provider=provider,
        status=status,
    ).observe(latency_ms / 1000)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
