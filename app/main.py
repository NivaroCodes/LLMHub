from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.clients.redis_client import close_redis, init_redis
from app.db.database import close_db, init_db

load_dotenv()

from app.api.endpoints import router as chat
from app.api.monitoring import router as monitoring
from app.api.openai_endpoints import router as openai_compatible
from app.middleware.monitoring import MonitoringMiddleware
from app.services.llm_services import get_llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("PostgreSQL database initialized")

    app.state.redis = await init_redis()
    print("Redis client initialized")

    # One shared LLMService instance; same object imported by both endpoint
    # modules. Attached to app.state so request handlers / future DI paths
    # can resolve it without a second instantiation.
    app.state.llm_service = get_llm_service()

    yield

    await app.state.llm_service.aclose()

    await close_redis()
    print("Redis client closed")

    await close_db()
    print("PostgreSQL pool closed")

app = FastAPI(lifespan=lifespan)
app.add_middleware(MonitoringMiddleware)
app.include_router(chat)
app.include_router(openai_compatible)
app.include_router(monitoring)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/hi")
async def great():
    return {"message": "Welcome!"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)
