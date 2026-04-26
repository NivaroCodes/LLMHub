from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, generate_latest

CHAT_REQUESTS_TOTAL = Counter(
    "llmhub_chat_requests_total",
    "Total number of chat requests handled by the API.",
    labelnames=("provider", "status", "cached", "fallback_used"),
)

CHAT_REQUEST_LATENCY_SECONDS = Histogram(
    "llmhub_chat_request_latency_seconds",
    "End-to-end latency of chat requests.",
    labelnames=("provider", "status"),
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)

EVENT_LOOP_LAG = Gauge(
    "llmhub_event_loop_lag_seconds",
    "Current event loop lag in seconds."
)

LATENCY_BREAKDOWN = Histogram(
    "llmhub_latency_breakdown_seconds",
    "Latency breakdown by stage.",
    labelnames=("stage",),
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10)
)

PROVIDER_LATENCY = Histogram(
    "llmhub_provider_latency_seconds",
    "Latency per provider call.",
    labelnames=("provider", "stage"),  # stage: connect, first_token, total
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 40)
)

REDIS_LATENCY = Histogram(
    "llmhub_redis_latency_seconds",
    "Latency of Redis operations.",
    labelnames=("operation",),
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1)
)


PROVIDER_STATE = Gauge(
    "llmhub_provider_state",
    "Current state of the provider (1=Healthy, 0=Degraded, -1=Banned)",
    labelnames=("provider",)
)

CIRCUIT_STATE = Gauge(
    "llmhub_circuit_state",
    "Current state of the circuit (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
    labelnames=("provider",)
)

CIRCUIT_OPEN_COUNT = Counter(
    "llmhub_circuit_open_total",
    "Total number of times a circuit has opened.",
    labelnames=("provider", "reason")
)

FALLBACK_LEVEL_USED = Counter(
    "llmhub_fallback_level_total",
    "Total number of times each fallback level was used.",
    labelnames=("level",) # levels: cache, race_winner, manual_fallback, static_safe
)

REQUEST_DROPPED_TOTAL = Counter(
    "llmhub_request_dropped_total",
    "Total number of requests dropped due to no available providers or overload.",
    labelnames=("reason",)
)

REQUEST_BLOCKED_BY_RATE_LIMIT = Counter(
    "llmhub_request_blocked_by_rate_limit_total",
    "Total number of requests blocked by provider-specific rate limiter.",
    labelnames=("provider",)
)

RETRY_BACKOFF_SECONDS = Counter(
    "llmhub_retry_backoff_seconds_total",
    "Total backoff time introduced by retries/429s.",
    labelnames=("provider",)
)

SYSTEM_STATE = Gauge(
    "llmhub_system_state",
    "Current system health state (0=Healthy, 1=Degraded, 2=Critical)"
)

ACTIVE_PROVIDERS_COUNT = Gauge(
    "llmhub_active_providers_count",
    "Number of currently active (healthy) providers"
)

TIME_IN_STATE = Counter(
    "llmhub_time_in_state_seconds_total",
    "Total time spent in each system state.",
    labelnames=("state",)
)

BLOCKED_REQUESTS_TOTAL = Counter(
    "llmhub_blocked_requests_total",
    "Total number of requests blocked by system control plane.",
    labelnames=("reason", "state")
)

PROVIDER_HEALTH_SCORE = Gauge(
    "llmhub_provider_health_score",
    "Health score of a provider (0-100)",
    labelnames=("provider",)
)

FALLBACK_REASON = Counter(
    "llmhub_fallback_reason_total",
    "Total number of times fallback was triggered by reason.",
    labelnames=("reason", "provider")
)

COOLDOWN_ACTIVE = Gauge(
    "llmhub_cooldown_active",
    "Whether global cooldown is currently active (1 or 0)"
)

PROVIDER_FAILURE_RATE = Counter(
    "llmhub_provider_failure_total",
    "Total number of provider failures.",
    labelnames=("provider", "error_type")
)

PROVIDER_SUCCESS_RATE = Counter(
    "llmhub_provider_success_total",
    "Total number of provider successes.",
    labelnames=("provider",)
)

COOLDOWN_TRIGGER_COUNT = Counter(
    "llmhub_cooldown_trigger_total",
    "Total number of times global cooldown was triggered."
)

REQUEST_RATE_PER_SECOND = Gauge(
    "llmhub_request_rate_per_second",
    "Calculated requests per second (from gateway perspective)."
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
