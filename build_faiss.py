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
        print(f"Loading FastEmbed model: {model_name}")
        self.model = TextEmbedding(model_name=model_name)
        print("FastEmbed model loaded successfully.")

    def embed_documents(self, texts):
        """Embed multiple documents"""
        return list(self.model.embed(texts))

    def embed_query(self, text):
        """Embed a single query"""
        return list(self.model.embed([text]))[0]


# -----------------------------
# Load and Split PDF
# -----------------------------
print("Loading PDF document...")
loader = PyPDFLoader("Final_Data.pdf")
docs = loader.load()
print(f"PDF loaded with {len(docs)} pages.")

print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
print(f"Document split into {len(chunks)} chunks.")


# -----------------------------
# Initialize Embedding Model
# -----------------------------
emb = FastEmbedWrapper(model_name="BAAI/bge-small-en")


# -----------------------------
# Build and Save FAISS Vector Store
# -----------------------------
print("Building FAISS vector store...")
vector_store = FAISS.from_documents(documents=chunks, embedding=emb)

print("Saving FAISS index to disk...")
vector_store.save_local("./faiss_vectorstore")

print(f"\n✓ Success!")
print(f"  - Total chunks: {len(chunks)}")
print(f"  - FAISS index saved to: ./faiss_vectorstore")