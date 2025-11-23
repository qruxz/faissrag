import asyncio
import os
from typing import Dict, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv
from groq import Groq

# Minimal imports - only what we need
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

# Load environment
load_dotenv()

# ==================== CONFIG ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FAISS_DIR = "faiss_vectorstore"
EMBED_MODEL = "BAAI/bge-small-en"

FALLBACK_MSG = "I am unable to understand your query or it is not relevant to the services we provide."
BANNED_WORDS = {"fuck", "fucker", "sex", "vagina", "dick", "pussy", "suicide", "terrorist"}


# ==================== FASTEMBEDDING WRAPPER ====================
class FastEmbedWrapper(Embeddings):
    def __init__(self, model_name=EMBED_MODEL):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# ==================== FAISS INITIALIZATION ====================
def build_faiss_index():
    """Build FAISS index from PDF"""
    print("📚 Building FAISS index from PDF...")
    
    if os.path.exists(FAISS_DIR):
        print(f"✅ FAISS index already exists at {FAISS_DIR}")
        return True
    
    try:
        # Load PDF
        loader = PyPDFLoader("Final_Data.pdf")
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages from PDF")
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        print(f"✅ Split into {len(chunks)} chunks")
        
        # Create embeddings and FAISS
        emb = FastEmbedWrapper(model_name=EMBED_MODEL)
        vector_store = FAISS.from_documents(documents=chunks, embedding=emb)
        vector_store.save_local(FAISS_DIR)
        print(f"✅ FAISS index saved to {FAISS_DIR}")
        return True
        
    except FileNotFoundError as e:
        print(f"⚠️  Final_Data.pdf not found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error building FAISS: {e}")
        return False


def load_faiss_index():
    """Load existing FAISS index"""
    try:
        emb = FastEmbedWrapper(model_name=EMBED_MODEL)
        vector_store = FAISS.load_local(
            FAISS_DIR,
            emb,
            allow_dangerous_deserialization=True
        )
        print(f"✅ FAISS index loaded from {FAISS_DIR}")
        return vector_store
    except Exception as e:
        print(f"❌ Error loading FAISS: {e}")
        return None


# ==================== GROQ LLM ====================
def groq_chat(messages, model="llama-3.1-8b-instant"):
    """Call Groq API"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return FALLBACK_MSG


# ==================== ABUSE FILTER ====================
def contains_banned(text: str) -> bool:
    """Check for banned words"""
    return any(w in text.lower() for w in BANNED_WORDS)


# ==================== RAG PIPELINE ====================
async def rag_chat(session_id: str, query: str, vector_store, session_memory: Dict):
    """Main RAG pipeline"""
    
    # Check abuse
    if contains_banned(query):
        return FALLBACK_MSG
    
    # Initialize session
    if session_id not in session_memory:
        session_memory[session_id] = []
    
    # Search FAISS
    try:
        results = await run_in_threadpool(
            vector_store.similarity_search_with_score,
            query,
            5
        )
        docs = [doc for doc, _ in results[:3]]
    except Exception as e:
        print(f"❌ FAISS search error: {e}")
        return FALLBACK_MSG
    
    if not docs:
        return FALLBACK_MSG
    
    # Build context
    context = "\n\n".join([d.page_content for d in docs])
    
    system_prompt = f"""You are a helpful assistant for Shyampari Edutech.
Use ONLY the information below to answer.
If answer not found, say: "{FALLBACK_MSG}"

Information:
{context}"""
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    messages += session_memory[session_id]
    messages.append({"role": "user", "content": query})
    
    # Call LLM
    try:
        answer = await run_in_threadpool(groq_chat, messages)
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return FALLBACK_MSG
    
    # Check abuse on response
    if contains_banned(answer):
        return FALLBACK_MSG
    
    # Save to memory
    session_memory[session_id].append({"role": "user", "content": query})
    session_memory[session_id].append({"role": "assistant", "content": answer})
    
    return answer


# ==================== FASTAPI APP ====================
app = FastAPI(title="Shyampari Edutech Chatbot")

# CORS - Allow all origins for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store = None
session_memory: Dict[str, List] = {}


# ==================== REQUEST MODELS ====================
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


# ==================== ROUTES ====================
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global vector_store
    
    print("\n" + "="*60)
    print("🚀 Starting Shyampari Edutech Chatbot...")
    print("="*60)
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Files in current directory: {os.listdir('.')[:10]}")
    
    # Check if FAISS exists
    if os.path.exists(FAISS_DIR):
        print(f"\n✅ FAISS directory found at {FAISS_DIR}")
        print(f"📁 Contents: {os.listdir(FAISS_DIR)}")
        vector_store = load_faiss_index()
    else:
        print(f"\n⚠️  FAISS directory not found at {FAISS_DIR}")
        
        # Try to build FAISS if PDF exists
        if os.path.exists("Final_Data.pdf"):
            print("📄 Found Final_Data.pdf, building FAISS index...")
            try:
                success = await run_in_threadpool(build_faiss_index)
                if success:
                    vector_store = load_faiss_index()
                else:
                    print("❌ Failed to build FAISS index")
                    vector_store = None
            except Exception as e:
                print(f"❌ Exception while building FAISS: {e}")
                vector_store = None
        else:
            print("❌ Final_Data.pdf not found in current directory!")
            print("📝 Please commit Final_Data.pdf to your repository")
            vector_store = None
    
    # Check GROQ API Key
    print("\n" + "-"*60)
    if GROQ_API_KEY:
        print("✅ GROQ_API_KEY configured")
    else:
        print("⚠️  GROQ_API_KEY not set in environment!")
    
    # Final status
    print("-"*60)
    if vector_store is None:
        print("❌ WARNING: FAISS not available!")
        print("   Chat will not work until FAISS is ready")
    else:
        print("✅ FAISS vector store ready!")
    
    print("="*60)
    print("✅ Startup complete!\n")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Shyampari Edutech Chatbot API",
        "version": "1.0",
        "endpoints": {
            "chat": "/chat",
            "api_chat": "/api/chat",
            "health": "/api/health"
        }
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint"""
    global vector_store, session_memory
    
    if vector_store is None:
        print(f"⚠️  Chat request but FAISS not ready: {request.message[:50]}")
        return ChatResponse(answer=FALLBACK_MSG)
    
    try:
        print(f"💬 Processing chat from session {request.session_id}")
        answer = await rag_chat(
            request.session_id,
            request.message,
            vector_store,
            session_memory
        )
        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        return ChatResponse(answer=FALLBACK_MSG)


@app.post("/api/chat")
async def api_chat_endpoint(request: ChatRequest) -> ChatResponse:
    """API chat endpoint (alternate)"""
    global vector_store, session_memory
    
    if vector_store is None:
        print(f"⚠️  API chat request but FAISS not ready: {request.message[:50]}")
        return ChatResponse(answer=FALLBACK_MSG)
    
    try:
        print(f"💬 Processing API chat from session {request.session_id}")
        answer = await rag_chat(
            request.session_id,
            request.message,
            vector_store,
            session_memory
        )
        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"❌ Error in API chat: {e}")
        return ChatResponse(answer=FALLBACK_MSG)


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Shyampari Edutech Chatbot",
        "faiss_ready": vector_store is not None,
        "groq_configured": GROQ_API_KEY is not None
    }


@app.get("/api/status")
async def status():
    """Detailed status endpoint"""
    return {
        "service": "Shyampari Edutech Chatbot",
        "version": "1.0",
        "faiss": {
            "status": "ready" if vector_store is not None else "not_ready",
            "path": FAISS_DIR,
            "exists": os.path.exists(FAISS_DIR)
        },
        "groq": {
            "status": "configured" if GROQ_API_KEY else "not_configured"
        },
        "sessions": len(session_memory)
    }


@app.post("/api/rebuild")
async def rebuild_endpoint():
    """Rebuild FAISS index endpoint"""
    global vector_store
    
    print("🔨 Rebuild request received...")
    
    if os.path.exists(FAISS_DIR):
        print("Removing existing FAISS index...")
        import shutil
        shutil.rmtree(FAISS_DIR)
    
    try:
        success = await run_in_threadpool(build_faiss_index)
        if success:
            vector_store = load_faiss_index()
            return {"status": "success", "message": "FAISS rebuilt successfully"}
        else:
            return {"status": "error", "message": "Failed to build FAISS"}
    except Exception as e:
        print(f"❌ Rebuild error: {e}")
        return {"status": "error", "message": str(e)}


# ==================== RENDER DEPLOYMENT ====================
# Render expects the app to be exported as 'app'

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    print(f"🌐 Starting on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
