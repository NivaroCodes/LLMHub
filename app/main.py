from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from app.clients.redis_client import close_redis, init_redis
from app.db.database import close_db, init_db

load_dotenv()

from app.api.endpoints import router as chat
from app.api.openai_endpoints import router as openai_compatible


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("SQLite database initialized")

    app.state.redis = await init_redis()
    print("Redis client initialized")

    yield

    await close_redis()
    print("Redis client closed")

    await close_db()
    print("PostgreSQL pool closed")

app = FastAPI(lifespan=lifespan)
app.include_router(chat)
app.include_router(openai_compatible)


@app.get("/hi")
async def great():
    return {"message": "Welcome!"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)
