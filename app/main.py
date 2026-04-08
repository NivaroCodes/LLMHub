import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.api.endpoints import router as chat  # noqa: E402


app = FastAPI()

app.include_router(chat)

@app.get("/hi")
async def great():
	return {"message": "Welcome!"}

if __name__ == "__main__":
	uvicorn.run("app.main:app", reload=True)
