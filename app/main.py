from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from app.clients.redis_client import close_redis, init_redis

load_dotenv()

from app.api.endpoints import router as chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await init_redis()
    print("Redis client initialized")

    yield

    await close_redis()
    print("Redis client closed")

app = FastAPI(lifespan=lifespan)
app.include_router(chat)


@app.get("/hi")
async def great():
    return {"message": "Welcome!"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)
