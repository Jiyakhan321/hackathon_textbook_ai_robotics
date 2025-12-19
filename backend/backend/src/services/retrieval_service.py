from typing import List, Tuple
from qdrant_client.http import models
from qdrant_client import models as qdrant_models
import numpy as np
from ..config.settings import settings
from ..config.vector_db import client, COLLECTION_NAME
from ..models.book_chunk import BookChunkBase


class RetrievalService:
    """
    Service to handle retrieval of relevant chunks from the vector database (Qdrant)
    """
    
    def __init__(self):
        self.top_k = settings.top_k_chunks
        self.threshold = settings.similarity_threshold
    
    def retrieve_relevant_chunks(self, query_embedding: List[float], context_mode: str = "full_book", filters: dict = None) -> List[Tuple[BookChunkBase, float]]:
        """
        Retrieve the top-k most relevant chunks based on the query embedding
        """
        try:
            # Prepare filters based on context mode
            search_filter = None
            if context_mode == "selected_text" and filters:
                # When in selected text mode, we only want chunks with the specific filters
                must_conditions = []
                if "source_reference" in filters:
                    must_conditions.append(
                        models.FieldCondition(
                            key="source_reference",
                            match=models.MatchValue(value=filters["source_reference"])
                        )
                    )
                if must_conditions:
                    search_filter = models.Filter(must=must_conditions)

            # Perform the search in Qdrant
            search_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=self.top_k,
                with_payload=True,
                with_vectors=False,
            )
            
            # Filter results based on similarity threshold
            filtered_results = []
            for hit in search_results:
                if hit.score >= self.threshold:
                    # Create a BookChunkBase object from the result
                    payload = hit.payload
                    chunk = BookChunkBase(
                        chunk_id=payload["chunk_id"],
                        content=payload["content"],
                        chapter_name=payload["chapter_name"],
                        section_title=payload["section_title"],
                        page_number=payload.get("page_number"),
                        source_reference=payload["source_reference"],
                        embedding_vector=[],  # We don't return the embedding vector to save space
                        created_at=payload["created_at"]
                    )
                    filtered_results.append((chunk, hit.score))
            
            return filtered_results
            
        except Exception as e:
            raise Exception(f"Error during retrieval: {str(e)}")
    
    def retrieve_chunks_by_ids(self, chunk_ids: List[str]) -> List[BookChunkBase]:
        """
        Retrieve specific chunks by their IDs
        """
        try:
            # Search for points with specific IDs
            results = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=chunk_ids,
                with_payload=True,
                with_vectors=False
            )
            
            chunks = []
            for result in results:
                payload = result.payload
                chunk = BookChunkBase(
                    chunk_id=payload["chunk_id"],
                    content=payload["content"],
                    chapter_name=payload["chapter_name"],
                    section_title=payload["section_title"],
                    page_number=payload.get("page_number"),
                    source_reference=payload["source_reference"],
                    embedding_vector=[],  # We don't return the embedding vector to save space
                    created_at=payload["created_at"]
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error retrieving chunks by IDs: {str(e)}")
    
    def add_chunk_to_db(self, chunk_id: str, content: str, embedding: List[float], metadata: dict) -> bool:
        """
        Add a single chunk to the vector database
        """
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    qdrant_models.PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload={
                            "content": content,
                            "chapter_name": metadata.get("chapter_name", ""),
                            "section_title": metadata.get("section_title", ""),
                            "page_number": metadata.get("page_number"),
                            "source_reference": metadata.get("source_reference", ""),
                            "created_at": metadata.get("created_at", "")
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"Error adding chunk to DB: {str(e)}")
            return False
    
    def add_chunks_to_db(self, chunks_data: List[dict]) -> bool:
        """
        Add multiple chunks to the vector database
        """
        try:
            points = []
            for chunk_data in chunks_data:
                point = qdrant_models.PointStruct(
                    id=chunk_data["chunk_id"],
                    vector=chunk_data["embedding"],
                    payload={
                        "content": chunk_data["content"],
                        "chapter_name": chunk_data["metadata"]["chapter_name"],
                        "section_title": chunk_data["metadata"]["section_title"],
                        "page_number": chunk_data["metadata"].get("page_number"),
                        "source_reference": chunk_data["metadata"]["source_reference"],
                        "created_at": chunk_data["metadata"].get("created_at", "")
                    }
                )
                points.append(point)
            
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            return True
        except Exception as e:
            print(f"Error adding chunks to DB: {str(e)}")
            return False