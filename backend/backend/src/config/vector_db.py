from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..config.settings import settings


# Initialize Qdrant client
client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    prefer_grpc=False  # Using HTTP for simplicity in this context
)


# Collection name for book chunks
COLLECTION_NAME = "book_chunks"


def initialize_qdrant_collection():
    """
    Initialize the Qdrant collection for storing book chunks with embeddings.
    This should be called during application startup.
    """
    # Check if collection already exists
    collections = client.get_collections()
    collection_names = [collection.name for collection in collections.collections]
    
    if COLLECTION_NAME not in collection_names:
        # Create collection with appropriate vector configuration
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1024,  # Cohere's multilingual embedding model returns 1024-dim vectors
                distance=models.Distance.COSINE
            )
        )
        
        # Create payload index for faster filtering by chapter and section
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="chapter_name",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="section_title",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        print(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        print(f"Qdrant collection {COLLECTION_NAME} already exists")