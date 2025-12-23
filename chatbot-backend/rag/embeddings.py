from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedding_model():
    """
    Get the OpenAI embedding model
    """
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def embed_text(text: str) -> list:
    """
    Create an embedding for a single text
    """
    embedding_model = get_embedding_model()
    return embedding_model.embed_query(text)

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Create embeddings for multiple texts
    """
    embedding_model = get_embedding_model()
    return await embedding_model.aembed(texts)