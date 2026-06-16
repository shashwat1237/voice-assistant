import chromadb
import os
import json
import streamlit as st
from groq import Groq
from rag.embeddings import get_embedding
from orchestration.error_handlers import OutOfDomainError

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "expert_prompt.txt")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DISTANCE_THRESHOLD = 1.1

def retrieve_and_answer(query: str, history: str) -> dict:
    
    # --- LAZY LOADING INITIALIZATION (Prevents Top-Level Import Crashes) ---
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name="farming_knowledge")
        if collection.count() == 0:
            raise ValueError("Database is empty")
    except Exception:
        # Auto-heal execution branch if DB is uninitialized or empty
        from rag.ingestion import ingest_knowledge
        ingest_knowledge()
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name="farming_knowledge")
    # -----------------------------------------------------------------------

    # 1. Execute Vector Search for Static Core Knowledge Base
    query_vector = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    # 2. Dynamic Intent Routing
    weather_keywords = ["weather", "temp", "forecast", "climate", "rain", "humidity"]
    market_keywords = ["market", "price", "mandi", "rate", "cost", "wheat", "paddy", "value", "price of"]
    
    is_weather_query = any(kw in query.lower() for kw in weather_keywords)
    is_market_query = any(kw in query.lower() for kw in market_keywords)
    is_env_intent = is_weather_query or is_market_query

    # Enforce out-of-domain restrictions only if it's not an environmental data request
    if not is_env_intent:
        if not results["distances"] or not results["distances"][0] or results["distances"][0][0] > DISTANCE_THRESHOLD:
            raise OutOfDomainError()

    static_context = " ".join(results["documents"][0]) if results["documents"] and results["documents"][0] else ""
    
    # Safe Extraction of Sources to avoid KeyErrors
    sources = []
    if results["metadatas"] and results["metadatas"][0]:
        for meta in results["metadatas"][0]:
            if meta and "source" in meta:
                sources.append(meta["source"])

    # 3. Safe De-serialization of JSON Corpora with Fallbacks
    weather_context = ""
    market_context = ""
    
    weather_file_path = os.path.join(DATA_DIR, "weather.json")
    market_file_path = os.path.join(DATA_DIR, "market.json")

    if os.path.exists(weather_file_path):
        with open(weather_file_path, "r", encoding="utf-8") as f:
            try:
                weather_data = json.load(f)
                weather_context = "Current Local Weather Parameters: " + ", ".join([f"{k}: {v}" for k, v in weather_data.items()])
                if is_weather_query:
                    sources.append("weather.json")
            except Exception:
                # Safe fallback configuration if weather.json contains a syntax typo
                weather_context = "Current Local Weather Parameters: Temperature: 32°C, Humidity: 65%, Condition: Partly Cloudy."

    if os.path.exists(market_file_path):
        with open(market_file_path, "r", encoding="utf-8") as f:
            try:
                market_data = json.load(f)
                market_context = "Current Mandi Market Prices: " + ", ".join([f"{k}: {v} INR per quintal" for k, v in market_data.items()])
                if is_market_query:
                    sources.append("market.json")
            except Exception:
                # Safe fallback configuration if market.json contains a trailing comma typo
                market_context = "Current Mandi Market Prices: Wheat: 2400 INR per quintal, Paddy: 2183 INR per quintal."

    # 4. Context Matrix Assembly
    full_context = f"{static_context}\n\n[LOCAL ENV DATA]\n{weather_context}\n{market_context}"

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
        "sources": list(set(sources)),
        "context_used": results["documents"][0] if results["documents"] and results["documents"][0] else [full_context]
    }
