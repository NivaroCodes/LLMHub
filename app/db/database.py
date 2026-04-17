import os
from datetime import datetime
from typing import Any

import asyncpg

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://llmhub:llmhub@localhost:5433/llmhub"
)

_pool: asyncpg.Pool | None = None

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS request_logs (
        id                BIGSERIAL    PRIMARY KEY,
        request_id        TEXT         NOT NULL,
        timestamp         TIMESTAMPTZ  NOT NULL,
        message           TEXT         NOT NULL,
        provider          TEXT         NOT NULL,
        model             TEXT         NOT NULL,
        latency_ms        INTEGER      NOT NULL,
        cached            BOOLEAN      NOT NULL,
        fallback_used     BOOLEAN      NOT NULL,
        prompt_tokens     INTEGER,
        completion_tokens INTEGER,
        cost_usd          NUMERIC(12, 8),
        status            TEXT         NOT NULL DEFAULT 'ok',
        error             TEXT
    )
"""


async def init_db() -> None:
    global _pool

    print(f"[DB] Connecting to: {_DATABASE_URL}")

    try:
        conn = await asyncpg.connect(
            _DATABASE_URL,
            ssl=None,
            timeout=10
        )

        await conn.execute("SELECT 1")
        await conn.close()

        _pool = await asyncpg.create_pool(
            _DATABASE_URL,
            min_size=1,
            max_size=5,
            ssl=None,
            command_timeout=10
        )

    except Exception as e:
        print("[DB] Connection failed:", e)
        raise

    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE)


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized")
    return _pool


async def log_request(
    request_id: str,
    timestamp: datetime,
    message: str,
    provider: str,
    model: str,
    latency_ms: int,
    cached: bool,
    fallback_used: bool,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cost_usd: float | None,
    status: str = "ok",
    error: str | None = None,
) -> None:
    async with _get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO request_logs
                (request_id, timestamp, message, provider, model, latency_ms,
                 cached, fallback_used, prompt_tokens, completion_tokens,
                 cost_usd, status, error)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
            request_id, timestamp, message, provider, model, latency_ms,
            cached, fallback_used, prompt_tokens, completion_tokens,
            cost_usd, status, error,
        )


async def get_stats() -> dict[str, Any]:
    async with _get_pool().acquire() as conn:
        all_time = await conn.fetchrow("""
            SELECT
                COUNT(*)                                        AS total,
                COUNT(*) FILTER (WHERE cached)                  AS cached_count,
                COALESCE(SUM(cost_usd) FILTER (WHERE NOT cached), 0) AS total_cost,
                COALESCE(AVG(latency_ms), 0)                    AS avg_latency,
                COUNT(*) FILTER (WHERE status = 'ok')           AS ok_count,
                COUNT(*) FILTER (WHERE status = 'error')        AS error_count
            FROM request_logs
        """)

        today = await conn.fetchrow("""
            SELECT
                COUNT(*)                                        AS total,
                COUNT(*) FILTER (WHERE cached)                  AS cached_count,
                COALESCE(SUM(cost_usd) FILTER (WHERE NOT cached), 0) AS total_cost,
                COALESCE(AVG(latency_ms), 0)                    AS avg_latency
            FROM request_logs
            WHERE timestamp::date = CURRENT_DATE
        """)

        by_provider = await conn.fetch("""
            SELECT
                provider,
                COUNT(*)                                             AS requests,
                COALESCE(SUM(cost_usd) FILTER (WHERE NOT cached), 0) AS cost_usd,
                COALESCE(AVG(latency_ms), 0)                         AS avg_latency_ms
            FROM request_logs
            WHERE status = 'ok'
            GROUP BY provider
            ORDER BY requests DESC
        """)

        savings_row = await conn.fetchrow("""
            SELECT
                (SELECT COALESCE(AVG(cost_usd), 0) FROM request_logs WHERE NOT cached AND status = 'ok')
                * (SELECT COUNT(*) FROM request_logs WHERE cached AND status = 'ok')
                AS savings
        """)

    total        = all_time["total"] or 0
    cached_count = all_time["cached_count"] or 0
    today_total  = today["total"] or 0
    today_cached = today["cached_count"] or 0

    return {
        "all_time": {
            "total_requests":  total,
            "cached_requests": cached_count,
            "cache_hit_rate":  round(cached_count / total, 4) if total else 0.0,
            "total_cost_usd":  round(float(all_time["total_cost"]), 6),
            "avg_latency_ms":  round(float(all_time["avg_latency"])),
            "ok_count":        all_time["ok_count"] or 0,
            "error_count":     all_time["error_count"] or 0,
        },
        "today": {
            "total_requests":  today_total,
            "cached_requests": today_cached,
            "cache_hit_rate":  round(today_cached / today_total, 4) if today_total else 0.0,
            "total_cost_usd":  round(float(today["total_cost"]), 6),
            "avg_latency_ms":  round(float(today["avg_latency"])),
        },
        "by_provider": [
            {
                "provider":       row["provider"],
                "requests":       row["requests"],
                "cost_usd":       round(float(row["cost_usd"]), 6),
                "avg_latency_ms": round(float(row["avg_latency_ms"])),
            }
            for row in by_provider
        ],
        "estimated_cache_savings_usd": round(float(savings_row["savings"] or 0), 6),
    }