import asyncio
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from rag_engine import rag_chat_startup, vector_store

# ---------------------
# Test Data
# ---------------------
test_questions = [
    "When was Shyampari Edutech officially registered under the Government of India?",
    "Which cities in India does Shyampari Edutech provide both online and offline tutoring services?",
    "Name three international locations where Shyampari Edutech offers tutoring services.",
    "What is the maximum number of students allowed in group tuition batches?",
    "Which educational boards does Shyampari Edutech support for tutoring?",
    "Does Shyampari Edutech provide coaching for competitive exams like JEE and NEET?",
    "What kind of support do coordinators offer to parents and students?",
    "What is the typical duration of a demo class?",
    "What steps must a student complete before attending a demo class?",
    "Within how many days does Shyampari Edutech finalize and share a tutor profile after demo registration?",
]

ground_truth_answers = [
    "Shyampari Edutech was officially registered under the Government of India in 2023.",
    "Shyampari Edutech provides both online and offline tutoring services in Pune and Mumbai.",
    "The company offers tutoring services in Dubai, Oman, Saudi Arabia, and Malaysia.",
    "Group tuition batches include 3 to 5 students.",
    "They support ICSE, IGCSE, CBSE, IB, and A-Level boards.",
    "Yes, they provide coaching for exams like JEE, NEET, NDA, Sainik School, and Rashtriya Military School.",
    "Coordinators monitor class schedules, punctuality, syllabus completion, and maintain communication between tutors and parents.",
    "A demo class typically lasts 20 to 30 minutes.",
    "Students must sign up, share their location on WhatsApp, fill the registration form, and pay the demo fee.",
    "The tutor profile is finalized and shared within 2–3 days after demo registration.",
]


# ---------------------
# Metrics Functions
# ---------------------
try:
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    OLLAMA_AVAILABLE = True
except:
    print("⚠️  Ollama not available. Skipping semantic similarity metrics.")
    OLLAMA_AVAILABLE = False


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def answer_similarity(answer, gt):
    """Semantic similarity between generated and ground truth answers"""
    if not OLLAMA_AVAILABLE:
        return None
    try:
        v1 = embed_model.embed_query(answer)
        v2 = embed_model.embed_query(gt)
        return cosine_similarity(v1, v2)
    except:
        return None


def context_recall(contexts, gt):
    """Measure how many key words from ground truth are in retrieved contexts"""
    text = " ".join(contexts).lower()
    words = [w for w in gt.lower().split() if len(w) > 3]
    found = sum(1 for w in words if w in text)
    return found / len(words) if words else 0


# ---------------------
# Evaluation Loop
# ---------------------
async def run_evaluation():
    """Run evaluation on test questions"""
    print("=" * 70)
    print("🧪 Starting RAG Evaluation")
    print("=" * 70)
    
    total_sim = 0
    total_recall = 0
    count = 0
    
    for idx, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"📝 Query {idx}/{len(test_questions)}: {question}")
        print(f"{'='*70}")
        
        try:
            # Retrieve documents
            docs = vector_store.similarity_search(question, k=8)
            contexts = [d.page_content for d in docs[:3]]
            
            # Generate answer
            answer = await rag_chat_startup(f"eval-session-{idx}", question)
            print(f"\n✓ Generated Answer:\n{answer}")
            
            # Calculate metrics
            sim = answer_similarity(answer, ground_truth_answers[idx - 1])
            recall = context_recall(contexts, ground_truth_answers[idx - 1])
            
            if sim is not None:
                print(f"\n📊 Metrics:")
                print(f"   - Answer Similarity: {sim:.3f}")
                total_sim += sim
            
            print(f"   - Context Recall:    {recall:.3f}")
            total_recall += recall
            
            count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            continue
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📈 Evaluation Summary")
    print(f"{'='*70}")
    print(f"Total Queries Evaluated: {count}")
    if OLLAMA_AVAILABLE:
        print(f"Average Answer Similarity: {total_sim/count:.3f}")
    print(f"Average Context Recall:    {total_recall/count:.3f}")
    print(f"{'='*70}\n")


# ---------------------
# Run
# ---------------------
if __name__ == "__main__":
    asyncio.run(run_evaluation())