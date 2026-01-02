from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Load the Qwen-based embedding model (using a similar model for now)
def get_embedding_model():
    """
    Get the Qwen-based embedding model
    """
    # Using a multilingual sentence transformer as a proxy for Qwen embeddings
    # In production, this would be replaced with the actual Qwen embedding model
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text: str) -> list:
    """
    Create an embedding for a single text
    """
    embedding_model = get_embedding_model()
    embedding = embedding_model.encode([text])
    return embedding[0].tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Create embeddings for multiple texts
    """
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(texts)
    return [embedding.tolist() for embedding in embeddings]