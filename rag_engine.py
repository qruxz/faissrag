import asyncio
import os
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from starlette.concurrency import run_in_threadpool

# FastEmbed
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from dotenv import load_dotenv
from groq import Groq

# ---------------------
# Configuration
# ---------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FAISS_DIR = "faiss_vectorstore"
EMBED_MODEL = "BAAI/bge-small-en"

FAISS_K = 8
FINAL_K = 3

FALLBACK_MSG = "I am unable to understand your query or it is not relevant to the services we provide."
BANNED_WORDS = {"fuck", "fucker", "sex", "vagina", "dick", "pussy", "suicide", "terrorist", "religion"}


# ---------------------
# FastEmbed Wrapper for LangChain
# ---------------------
class FastEmbedWrapper(Embeddings):
    def __init__(self, model_name=EMBED_MODEL):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# ---------------------
# Load Vector Store
# ---------------------
print("Initializing RAG engine...")
emb = FastEmbedWrapper(model_name=EMBED_MODEL)
try:
    vector_store = FAISS.load_local(
        FAISS_DIR,
        emb,
        allow_dangerous_deserialization=True
    )
    print(f"✓ FAISS vector store loaded from {FAISS_DIR}")
except Exception as e:
    print(f"✗ Error loading FAISS: {e}")
    print(f"Please run build_faiss.py first to create the index.")
    raise

SESSION_MEMORY: Dict[str, List[Dict]] = {}


# ---------------------
# Abuse Filter
# ---------------------
def contains_banned(text: str) -> bool:
    """Check if text contains banned words"""
    low = text.lower()
    return any(w in low for w in BANNED_WORDS)


# ---------------------
# Groq LLM
# ---------------------
def groq_chat(messages, model="llama-3.1-8b-instant"):
    """Call Groq API for chat completion"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq: {e}")
        return FALLBACK_MSG


# ---------------------
# RAG Chat Pipeline
# ---------------------
async def rag_chat_startup(session_id: str, query: str):
    """
    Main RAG pipeline for answering user queries.
    Uses FAISS for retrieval and Groq LLM for generation.
    """
    
    # Initialize session memory if needed
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    # Abuse filter on user input
    if contains_banned(query):
        return FALLBACK_MSG

    # 1) FAISS retrieval
    try:
        results = await run_in_threadpool(
            vector_store.similarity_search_with_score,
            query,
            FAISS_K
        )
        faiss_docs = [doc for doc, _ in results]
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return FALLBACK_MSG

    # 2) Select top-k documents
    faiss_docs = faiss_docs[:FINAL_K]

    if not faiss_docs:
        return FALLBACK_MSG

    # 3) Build context from retrieved documents
    context = "\n\n".join([d.page_content for d in faiss_docs])

    system_prompt = f"""You are a helpful assistant for Shyampari Edutech. 
Use ONLY the information provided below to answer the user's question.
If the answer is not in the provided information, respond with: "{FALLBACK_MSG}"

Information:
{context}
"""

    # 4) Build message history
    messages = [{"role": "system", "content": system_prompt}]
    messages += SESSION_MEMORY[session_id]
    messages.append({"role": "user", "content": query})

    # 5) Call Groq LLM
    try:
        answer = await run_in_threadpool(groq_chat, messages)
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return FALLBACK_MSG

    # 6) Abuse filter on response
    if contains_banned(answer):
        return FALLBACK_MSG

    # 7) Save to session memory
    SESSION_MEMORY[session_id].append({"role": "user", "content": query})
    SESSION_MEMORY[session_id].append({"role": "assistant", "content": answer})

    return answer