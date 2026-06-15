import chromadb
import os
import google.generativeai as genai
from rag.embeddings import get_embedding
from orchestration.error_handlers import OutOfDomainError

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "expert_prompt.txt")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
DISTANCE_THRESHOLD = 1.1

def retrieve_and_answer(query: str, history: str) -> dict:
    
    # --- AUTO-HEALING DATABASE LOGIC ---
    try:
        collection = chroma_client.get_collection(name="farming_knowledge")
        if collection.count() == 0:
            raise ValueError("Database is empty")
    except Exception:
        from rag.ingestion import ingest_knowledge
        ingest_knowledge()
        collection = chroma_client.get_collection(name="farming_knowledge")
    # -----------------------------------

    query_vector = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    if not results["distances"] or not results["distances"][0] or results["distances"][0][0] > DISTANCE_THRESHOLD:
        raise OutOfDomainError()

    context = " ".join(results["documents"][0])
    sources = list(set([meta["source"] for meta in results["metadatas"][0]]))

    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read()

    prompt = system_prompt.replace("{history}", history).replace("{context}", context)
    prompt += f"\n\nQuestion: {query}\nAnswer:"

    # --- GEMINI INTEGRATION ---
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    return {
        "answer": response.text.strip(),
        "sources": sources,
        "context_used": results["documents"][0]
    }
