from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


# -----------------------------
# FastEmbed Wrapper for LangChain
# -----------------------------
class FastEmbedWrapper(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        # returns generator → convert to list
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


# -----------------------------
# Load and Split PDF
# -----------------------------
loader = PyPDFLoader("Final_Data.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)


# -----------------------------
# Embedding Model (FastEmbed)
# -----------------------------
emb = FastEmbedWrapper(model_name="BAAI/bge-small-en")


# -----------------------------
# Build FAISS Vector Store
# -----------------------------
vector_store = FAISS.from_documents(documents=chunks, embedding=emb)
vector_store.save_local("./faiss_vectorstore")

print(f"Length of chunks: {len(chunks)}")
print("✓ FAISS index built and saved successfully.")
