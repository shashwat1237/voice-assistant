import chromadb
from huggingface_hub import InferenceClient
import os
from rag.embeddings import get_embedding
from orchestration.error_handlers import OutOfDomainError

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "expert_prompt.txt")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
hf_client = InferenceClient(token=os.environ.get("HF_TOKEN"), timeout=15.0)

# Patched: Tightened boundary to 1.1 ensures mathematical cosine overlap
DISTANCE_THRESHOLD = 1.1

def retrieve_and_answer(query: str, history: str) -> dict:
    collection = chroma_client.get_collection(name="farming_knowledge")
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

    # Patched: Removed invalid timeout kwarg
    answer = hf_client.text_generation(
        prompt,
        model="mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=300
    )

    return {
        "answer": answer.strip(),
        "sources": sources,
        "context_used": results["documents"][0]
    }
