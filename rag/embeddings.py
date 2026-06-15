import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en")

def get_embedding(text: str) -> list:
    embedding_model = load_embedding_model()
    # Explicit normalization forces vectors onto the unit hypersphere,
    # ensuring L2 distances mathematically align with the 1.1 threshold.
    return embedding_model.encode(text, normalize_embeddings=True).tolist()
