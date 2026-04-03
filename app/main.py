import uvicorn
from fastapi import FastAPI
from app.api.endpoints import router as chat


app = FastAPI()

app.include_router(chat)

@app.get("/hi")
async def great():
	return {"message": "Welcome!"}

if __name__ == "__main__":
	uvicorn.run("app.main:app", reload=True)
