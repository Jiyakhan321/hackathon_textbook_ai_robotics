from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_qdrant_client():
    """
    Get a Qdrant client instance
    """
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False  # Using HTTP for compatibility
    )
    return client