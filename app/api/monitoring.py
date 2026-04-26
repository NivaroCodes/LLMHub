from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.db.database import (
    get_monitoring_overview,
    get_monitoring_timeseries,
    get_recent_failures,
)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@router.get("/overview")
async def monitoring_overview(
    window_minutes: int = Query(60, ge=1, le=1440),
):
    try:
        return await get_monitoring_overview(window_minutes=window_minutes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Database is not initialized") from exc


@router.get("/timeseries")
async def monitoring_timeseries(
    window_minutes: int = Query(180, ge=5, le=1440),
    bucket_minutes: int = Query(5, ge=1, le=60),
):
    try:
        points = await get_monitoring_timeseries(
            window_minutes=window_minutes,
            bucket_minutes=bucket_minutes,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Database is not initialized") from exc
    return {
        "window_minutes": window_minutes,
        "bucket_minutes": bucket_minutes,
        "points": points,
    }


@router.get("/failures")
async def monitoring_failures(limit: int = Query(25, ge=1, le=200)):
    try:
        rows = await get_recent_failures(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Database is not initialized") from exc
    return {"count": len(rows), "items": rows}


@router.get("/dashboard", include_in_schema=False)
async def monitoring_dashboard():
    return FileResponse(_STATIC_DIR / "dashboard.html")
