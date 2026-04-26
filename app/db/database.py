import asyncio
import os
from datetime import datetime
from typing import Any

import asyncpg

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://llmhub:llmhub@localhost:5433/llmhub"
)

_pool: asyncpg.Pool | None = None
_queue: asyncio.Queue | None = None
_worker_task: asyncio.Task | None = None

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
    global _pool, _queue, _worker_task

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

    _queue = asyncio.Queue(maxsize=1000)
    _worker_task = asyncio.create_task(_db_worker())


async def _db_worker():
    from app.metrics import LATENCY_BREAKDOWN
    import time
    
    while True:
        try:
            item = await _queue.get()
            if item is None:
                break
            
            start_db = time.monotonic()
            try:
                async with _get_pool().acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO request_logs
                            (request_id, timestamp, message, provider, model, latency_ms,
                             cached, fallback_used, prompt_tokens, completion_tokens,
                             cost_usd, status, error)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        """,
                        *item
                    )
            except Exception as e:
                print(f"[DB] Worker error: {e}")
            finally:
                _queue.task_done()
                LATENCY_BREAKDOWN.labels(stage="db_log").observe(time.monotonic() - start_db)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[DB] Worker loop error: {e}")
            await asyncio.sleep(1)


async def close_db() -> None:
    global _pool, _queue, _worker_task
    if _queue:
        await _queue.put(None)
    if _worker_task:
        try:
            await asyncio.wait_for(_worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            _worker_task.cancel()
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
    if _queue is None:
        return
    
    try:
        _queue.put_nowait((
            request_id, timestamp, message, provider, model, latency_ms,
            cached, fallback_used, prompt_tokens, completion_tokens,
            cost_usd, status, error
        ))
    except asyncio.QueueFull:
        print("[DB] Queue full, dropping log")


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


async def get_monitoring_overview(window_minutes: int = 60) -> dict[str, Any]:
    window_minutes = max(1, min(window_minutes, 24 * 60))
    async with _get_pool().acquire() as conn:
        summary = await conn.fetchrow(
            """
            SELECT
                COUNT(*) AS total_requests,
                COUNT(*) FILTER (WHERE status = 'ok') AS ok_requests,
                COUNT(*) FILTER (WHERE status = 'error') AS error_requests,
                COUNT(*) FILTER (WHERE cached) AS cached_requests,
                COUNT(*) FILTER (WHERE fallback_used) AS fallback_requests,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS p95_latency_ms,
                COALESCE(SUM(cost_usd) FILTER (WHERE NOT cached), 0) AS cost_usd
            FROM request_logs
            WHERE timestamp >= NOW() - make_interval(mins => $1)
            """,
            window_minutes,
        )

        provider_rows = await conn.fetch(
            """
            SELECT
                provider,
                COUNT(*) AS requests,
                COUNT(*) FILTER (WHERE status = 'error') AS errors,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(SUM(cost_usd) FILTER (WHERE NOT cached), 0) AS cost_usd
            FROM request_logs
            WHERE timestamp >= NOW() - make_interval(mins => $1)
            GROUP BY provider
            ORDER BY requests DESC
            """,
            window_minutes,
        )

    total_requests = int(summary["total_requests"] or 0)
    ok_requests = int(summary["ok_requests"] or 0)
    error_requests = int(summary["error_requests"] or 0)
    cached_requests = int(summary["cached_requests"] or 0)
    fallback_requests = int(summary["fallback_requests"] or 0)

    return {
        "window_minutes": window_minutes,
        "kpis": {
            "total_requests": total_requests,
            "ok_requests": ok_requests,
            "error_requests": error_requests,
            "success_rate": round(ok_requests / total_requests, 4) if total_requests else 0.0,
            "error_rate": round(error_requests / total_requests, 4) if total_requests else 0.0,
            "cache_hit_rate": round(cached_requests / total_requests, 4) if total_requests else 0.0,
            "fallback_rate": round(fallback_requests / total_requests, 4) if total_requests else 0.0,
            "avg_latency_ms": round(float(summary["avg_latency_ms"] or 0)),
            "p95_latency_ms": round(float(summary["p95_latency_ms"] or 0)),
            "cost_usd": round(float(summary["cost_usd"] or 0), 6),
        },
        "providers": [
            {
                "provider": row["provider"],
                "requests": int(row["requests"] or 0),
                "errors": int(row["errors"] or 0),
                "avg_latency_ms": round(float(row["avg_latency_ms"] or 0)),
                "cost_usd": round(float(row["cost_usd"] or 0), 6),
            }
            for row in provider_rows
        ],
    }


async def get_monitoring_timeseries(
    window_minutes: int = 120,
    bucket_minutes: int = 5,
) -> list[dict[str, Any]]:
    window_minutes = max(5, min(window_minutes, 24 * 60))
    bucket_minutes = max(1, min(bucket_minutes, 60))

    async with _get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                date_bin(make_interval(mins => $1), timestamp, TIMESTAMPTZ '2000-01-01') AS bucket_start,
                COUNT(*) AS requests,
                COUNT(*) FILTER (WHERE status = 'error') AS errors,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS p95_latency_ms
            FROM request_logs
            WHERE timestamp >= NOW() - make_interval(mins => $2)
            GROUP BY bucket_start
            ORDER BY bucket_start ASC
            """,
            bucket_minutes,
            window_minutes,
        )

    return [
        {
            "bucket_start": row["bucket_start"].isoformat(),
            "requests": int(row["requests"] or 0),
            "errors": int(row["errors"] or 0),
            "avg_latency_ms": round(float(row["avg_latency_ms"] or 0)),
            "p95_latency_ms": round(float(row["p95_latency_ms"] or 0)),
        }
        for row in rows
    ]


async def get_recent_failures(limit: int = 25) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    async with _get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT timestamp, request_id, provider, model, status, error, latency_ms
            FROM request_logs
            WHERE status = 'error'
            ORDER BY timestamp DESC
            LIMIT $1
            """,
            limit,
        )

    return [
        {
            "timestamp": row["timestamp"].isoformat(),
            "request_id": row["request_id"],
            "provider": row["provider"],
            "model": row["model"],
            "status": row["status"],
            "error": row["error"] or "",
            "latency_ms": int(row["latency_ms"] or 0),
        }
        for row in rows
    ]
