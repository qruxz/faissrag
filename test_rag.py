# ------------------------------------------------------------
# 1. Define your test questions and ground-truth answers here
# ------------------------------------------------------------
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
    "Does Shyampari Edutech provide replacement tutors if the original tutor leaves mid-term?",
    "What materials are included in Shyampari Edutech’s complete study support?",
    "What is the recommended batch size for group tuition?",
    "What details are required from students during the sign-up process?",
    "What happens if a user enters incorrect or spam-like information during sign-up?",
    "What factors does the recommendation system use to match tutors and students?",
    "What is the average tuition fee range for senior secondary students?",
    "Are motivational workshops part of Shyampari Edutech’s services?",
    "What is the purpose of the demo registration fee?",
    "How long does a student’s demo class typically last?",
    "Does Shyampari Edutech offer language learning classes such as German or French?",
    "What are tutors expected to demonstrate during a demo session?",
    "What kind of academic monitoring is provided as part of counselling support?",
    "How can students or parents contact Shyampari Edutech for enrollment?",
    "What tools are used during demo classes for online sessions?",
    "What is the process for finalizing a tutor after the demo session?",
    "What type of foreign countries does Shyampari Edutech currently operate in?",
    "What is the tutor replacement policy if a tutor leaves unexpectedly?",
    "Does the company provide both academic and co-curricular tutoring?",
    "What must students do after the demo class to proceed with regular classes?"
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
    "Yes, the company provides replacement tutors within two weeks if a tutor leaves or if parents want a change.",
    "Study support includes notes, worksheets, practice papers, and test materials.",
    "The recommended batch size for group tuition is 3 to 5 students.",
    "Students must provide basic details, subjects, academic level, learning goals, and preferred class timings.",
    "Incorrect or spam-like information leads to account flagging and possible suspension if not corrected within 72 hours.",
    "The recommendation system uses subject, learning goals, availability, budget, ratings, reviews, demo outcomes, and engagement signals.",
    "The average tuition fee for senior secondary students is ₹10,000 to ₹15,000, with a maximum of ₹20,000.",
    "Yes, motivational workshops focusing on study skills and exam strategies are offered.",
    "The demo registration fee ensures serious enquiries and secures the demo booking.",
    "A demo class typically lasts between 20 and 30 minutes.",
    "Yes, they offer language classes such as German, French, and Spanish.",
    "Tutors introduce their teaching approach, give a mini-lesson, discuss goals, and clarify expectations.",
    "Academic monitoring includes test analysis, identifying mistakes, reviewing behavior, and checking homework and revision habits.",
    "Students or parents can contact via the website, email, or WhatsApp at 6299559291.",
    "Demo classes use tools like video call, chat, screen sharing, and whiteboard.",
    "After the demo, students provide feedback and finalize the tutor by selecting a plan and making the payment.",
    "They operate in countries such as Dubai, Oman, Saudi Arabia, and Malaysia and plan expansion to Singapore, Japan, and Australia.",
    "If a tutor leaves unexpectedly, a new tutor is assigned within two weeks.",
    "Yes, they provide academic tutoring, co-curricular training, and language learning.",
    "Students must share feedback and finalize the tutor by choosing and paying for a plan."
]

import asyncio
import numpy as np
from langchain_ollama import OllamaEmbeddings
from rag_engine import rag_chat_startup, reranker, vector_store


embed_model = OllamaEmbeddings(model="nomic-embed-text")


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def answer_similarity(answer, gt):
    v1 = embed_model.embed_query(answer)
    v2 = embed_model.embed_query(gt)
    return cosine_sim(v1, v2)


def context_recall(contexts, gt):
    text = " ".join(contexts).lower()
    words = [w for w in gt.lower().split() if len(w) > 3]
    found = sum(1 for w in words if w in text)
    return found / len(words)


# ======================
# Test questions
# ======================



async def run_eval():
    for idx, q in enumerate(test_questions):
        print(f"\n--- Query {idx+1}: {q}")

        # retrieve
        docs = vector_store.similarity_search(q, k=8)
        reranked = reranker.rerank(q, docs, top_k=3)
        contexts = [d.page_content for d in reranked]

        # answer
        ans = await rag_chat_startup(f"s-{idx}", q)
        print("Answer:", ans)

        # metrics
        sim = answer_similarity(ans, ground_truth_answers[idx])
        rec = context_recall(contexts, ground_truth_answers[idx])

        print(f"Answer Similarity: {sim:.3f}")
        print(f"Context Recall:    {rec:.3f}")


asyncio.run(run_eval())
