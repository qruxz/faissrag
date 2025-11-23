# # ---------------------
# # Reranker (FastEmbed ONNX)
# # ---------------------
# class SimpleReranker:
#     def __init__(self, model_name="cohere"):
#         print(f"Loading FastEmbed reranker: {model_name}")
#         self.model = Reranker(model_name=model_name)
#         print("Reranker loaded.")

#     def rerank(self, query: str, docs: List[Document], top_k: int):
#         if not docs:
#             return []

#         texts = [d.page_content for d in docs]

#         # Returns list of scores
#         scores = self.model.rerank(query, texts)

#         ranked = list(zip(docs, scores))
#         ranked.sort(key=lambda x: x[1], reverse=True)

#         return [d for d, _ in ranked[:top_k]]

# reranker = SimpleReranker()

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
# Config
# ---------------------

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FAISS_DIR = "faiss_vectorstore"
EMBED_MODEL = "BAAI/bge-small-en"

FAISS_K = 8
FINAL_K = 3

FALLBACK_MSG = "I am unable to understand your query or it is not relevant to the services we provide."
BANNED_WORDS = {"fuck", "fucker", "sex", "vagina", "dick", "pussy", "suicide", "terrorist", "relegion"}


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
# Load vectorstore
# ---------------------
emb = FastEmbedWrapper(model_name=EMBED_MODEL)
vector_store = FAISS.load_local(
    FAISS_DIR,
    emb,
    allow_dangerous_deserialization=True
)

SESSION_MEMORY: Dict[str, List[Dict]] = {}


# ---------------------
# Abuse filter
# ---------------------
def contains_banned(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in BANNED_WORDS)


# ---------------------
# Groq LLM
# ---------------------
def groq_chat(messages, model="llama-3.1-8b-instant"):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


# ---------------------
# FINAL RAG PIPELINE (no reranker)
# ---------------------
async def rag_chat_startup(session_id: str, query: str):

    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    # 1) FAISS search only
    results = await run_in_threadpool(
        vector_store.similarity_search_with_score,
        query,
        FAISS_K
    )
    faiss_docs = [doc for doc, _ in results]

    # 2) Select top-k directly from FAISS (no reranker)
    faiss_docs = faiss_docs[:FINAL_K]

    if not faiss_docs:
        return FALLBACK_MSG

    # 3) Build context
    context = "\n\n".join([d.page_content for d in faiss_docs])

    system_prompt = f"""
Use ONLY the information below to answer the user.
If the answer is not present, reply exactly:
"{FALLBACK_MSG}"

Information:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += SESSION_MEMORY[session_id]
    messages.append({"role": "user", "content": query})

    # 4) Call LLM
    answer = await run_in_threadpool(groq_chat, messages)

    # 5) Abuse filter
    if contains_banned(answer):
        return FALLBACK_MSG

    # 6) Save session memory
    SESSION_MEMORY[session_id].append({"role": "user", "content": query})
    SESSION_MEMORY[session_id].append({"role": "assistant", "content": answer})

    return answer
