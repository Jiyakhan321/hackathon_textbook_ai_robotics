from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging
from ..config.settings import settings
from ..utils.circuit_breaker import qdrant_circuit_breaker


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_link,
            api_key=settings.qdrant_api_key,
            timeout=10.0
        )
        self.collection_name = "book_content_embeddings"
        self._initialized = False

    async def initialize(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection for storing embeddings
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1024,  # Default size, adjust based on Cohere embedding dimensions
                        distance=models.Distance.COSINE
                    )
                )
                logging.info(f"Created Qdrant collection: {self.collection_name}")

            self._initialized = True
            logging.info("Qdrant service initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant service: {e}")
            raise

    @qdrant_circuit_breaker
    async def store_embedding(self,
                            content_id: UUID,
                            vector: List[float],
                            payload: Dict[str, Any]) -> str:
        """Store an embedding in Qdrant."""
        if not self._initialized:
            await self.initialize()

        point_id = str(content_id)

        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return point_id
        except Exception as e:
            logging.error(f"Failed to store embedding: {e}")
            raise

    @qdrant_circuit_breaker
    async def search_similar(self,
                           query_vector: List[float],
                           limit: int = 5,
                           filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            # Build filters if provided
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                if conditions:
                    search_filter = models.Filter(must=conditions)

            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=search_filter
            )

            return [
                {
                    "id": result.id,
                    "payload": result.payload,
                    "score": result.score
                }
                for result in results
            ]
        except Exception as e:
            logging.error(f"Failed to search similar embeddings: {e}")
            raise

    @qdrant_circuit_breaker
    async def delete_embedding(self, content_id: UUID) -> bool:
        """Delete an embedding from Qdrant."""
        if not self._initialized:
            await self.initialize()

        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[str(content_id)]
                )
            )
            return True
        except Exception as e:
            logging.error(f"Failed to delete embedding: {e}")
            return False

    async def close(self):
        """Close the Qdrant client connection."""
        if hasattr(self.client, '_client'):
            # QdrantClient doesn't have a close method, but we can clean up if needed
            pass


# Global instance
qdrant_service = QdrantService()