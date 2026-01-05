import cohere
from typing import List, Optional, Dict, Any
import logging
import asyncio
from ..config.settings import settings
from ..utils.circuit_breaker import cohere_circuit_breaker


class CohereService:
    def __init__(self):
        self.client = cohere.Client(settings.cohere_api_key)
        self._initialized = False

    async def initialize(self):
        """Initialize the Cohere service."""
        try:
            # Test the connection by making a simple call
            response = self.client.generate(
                model='command',
                prompt='Hello',
                max_tokens=5
            )
            self._initialized = True
            logging.info("Cohere service initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Cohere service: {e}")
            raise

    @cohere_circuit_breaker
    async def generate_embeddings(self, texts: List[str], model: str = "embed-english-v3.0") -> List[List[float]]:
        """Generate embeddings for the given texts."""
        if not self._initialized:
            await self.initialize()

        try:
            response = self.client.embed(
                texts=texts,
                model=model,
                input_type="search_document"  # Use search_document for content to be searched
            )
            return [embedding for embedding in response.embeddings]
        except Exception as e:
            logging.error(f"Failed to generate embeddings: {e}")
            raise

    @cohere_circuit_breaker
    async def generate_response(self, prompt: str, model: str = "command-r-plus") -> str:
        """Generate a response to the given prompt."""
        if not self._initialized:
            await self.initialize()

        try:
            response = self.client.generate(
                model=model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            return response.generations[0].text
        except Exception as e:
            logging.error(f"Failed to generate response: {e}")
            raise

    @cohere_circuit_breaker
    async def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to the query."""
        if not self._initialized:
            await self.initialize()

        try:
            response = self.client.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=documents,
                top_n=top_n
            )
            return [
                {
                    "index": rank.index,
                    "document": rank.document,
                    "relevance_score": rank.relevance_score
                }
                for rank in response.results
            ]
        except Exception as e:
            logging.error(f"Failed to rerank documents: {e}")
            raise


# Global instance
cohere_service = CohereService()