from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from rag_engine import rag_chat_startup

# ---------------------
# FastAPI Setup
# ---------------------
app = FastAPI(title="Shyampari Edutech Chatbot API")

# ---------------------
# CORS Configuration
# ---------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------
# Request/Response Models
# ---------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = None  # Optional, generated if not provided


class ChatResponse(BaseModel):
    response: str
    session_id: str
    success: bool = True


# ---------------------
# Routes
# ---------------------

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Shyampari Edutech Chatbot API",
        "version": "1.0"
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    Accepts user message and optional session_id.
    Returns AI response with session_id for conversation tracking.
    """
    try:
        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Call RAG pipeline
        response = await rag_chat_startup(session_id, request.message)
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            success=True
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return ChatResponse(
            response="Sorry, I encountered an error. Please try again later.",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False
        )


@app.get("/api/health")
async def health_check():
    """Health check with detailed status"""
    return {
        "status": "healthy",
        "service": "Shyampari Edutech Chatbot",
        "api_version": "1.0"
    }


# ---------------------
# Run Server
# ---------------------
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Shyampari Edutech Chatbot API...")
    print("📍 Server running at http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )