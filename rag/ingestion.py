import os
import chromadb
from rag.embeddings import get_embedding

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

def chunk_text(text: str, word_limit: int = 500, overlap: int = 100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), word_limit - overlap):
        chunks.append(" ".join(words[i:i + word_limit]))
    return chunks

def ingest_knowledge():
    collection = chroma_client.get_or_create_collection(name="farming_knowledge", metadata={"hnsw:space": "l2"})

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    files = [f for f in os.listdir(data_dir) if f.endswith(".md")]

    for file in files:
        with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            # Patched: Upsert ensures pipeline idempotency and prevents IDAlreadyExistsError
            collection.upsert(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": file}],
                ids=[f"{file}_chunk_{idx}"]
            )
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_knowledge()
