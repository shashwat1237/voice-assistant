import chromadb
import os
import streamlit as st
from groq import Groq
from rag.embeddings import get_embedding
from orchestration.error_handlers import OutOfDomainError

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "expert_prompt.txt")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
DISTANCE_THRESHOLD = 1.1

def get_mock_weather_and_market():
    """
    Plain-text runtime environmental layer. No JSON brackets to confuse the model.
    """
    mock_weather = "The current weather in Maharashtra is 32°C and partly cloudy."
    mock_market = (
        "The current market price for Wheat (गेहूं) is 2400 rupees per quintal. "
        "The current market price for Paddy (धान) is 2183 rupees per quintal. "
        "The current market price for Gram (चना) is 5440 rupees per quintal."
    )
    return mock_weather, mock_market

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

    static_context = " ".join(results["documents"][0])
    sources = list(set([meta["source"] for meta in results["metadatas"][0]]))

    # 1. Fetch the actual numerical parameters at runtime
    weather_context, market_context = get_mock_weather_and_market()

    # 2. Append them explicitly into the context tracking window
    full_context = (
        f"{static_context}\n\n"
        f"[LOCAL ENV DATA]\n"
        f"{weather_context}\n"
        f"{market_context}"
    )
    
    # Ensure our UI tracks the display label cleanly
    if "Mock Mandi API" not in sources:
        sources.append("Mock Mandi API")

    with open(PROMPT_PATH, "r") as f:
        system_prompt = f.read()

    prompt = system_prompt.replace("{history}", history).replace("{context}", full_context)

    # --- GROQ LPU INFERENCE ---
    api_key = st.secrets.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "system", "content": "ABSOLUTE RULE: Answer in a polite, conversational paragraph. You are strictly forbidden from outputting JSON, dictionaries, curly brackets {}, or the word 'API'."},
            {"role": "user", "content": f"Question: {query}"}
        ],
        temperature=0.3
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources,
        "context_used": results["documents"][0]
    }
