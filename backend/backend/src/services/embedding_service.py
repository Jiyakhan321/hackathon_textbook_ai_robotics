import cohere
from typing import List, Union
import numpy as np
from ..config.settings import settings
from ..models.book_chunk import BookChunkCreate


class EmbeddingService:
    """
    Service to handle text embedding using Cohere's embedding model
    """
    
    def __init__(self):
        self.client = cohere.Client(settings.cohere_api_key)
        self.model = settings.cohere_embedding_model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"  # Using search_document for knowledge base content
            )
            # Return the embedding vector as a list of floats
            return response.embeddings[0]
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"  # Using search_document for knowledge base content
            )
            # Return the embedding vectors as a list of lists of floats
            return [embedding for embedding in response.embeddings]
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def embed_book_chunks(self, chunks: List[BookChunkCreate]) -> List[BookChunkCreate]:
        """
        Add embeddings to book chunks
        """
        if not chunks:
            return chunks
            
        # Extract the text content from all chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings for all texts at once
        embeddings = self.embed_texts(texts)
        
        # Update each chunk with its embedding
        updated_chunks = []
        for i, chunk in enumerate(chunks):
            updated_chunk = chunk.copy(update={"embedding_vector": embeddings[i]})
            updated_chunks.append(updated_chunk)
        
        return updated_chunks
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (using search_query input type)
        """
        try:
            response = self.client.embed(
                texts=[query],
                model=self.model,
                input_type="search_query"  # Using search_query for user queries
            )
            # Return the embedding vector as a list of floats
            return response.embeddings[0]
        except Exception as e:
            raise Exception(f"Error generating query embedding: {str(e)}")