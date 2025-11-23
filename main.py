from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import rag_chat_startup

app = FastAPI()

# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat_api(request: Query):
    answer = await rag_chat_startup(request.session_id, request.message)
    return {"answer": answer}

@app.get("/")
def root():
    return {"status": "ok"}
