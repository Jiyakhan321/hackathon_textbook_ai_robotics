from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime
import logging
import asyncio
from ..models.query import Query
from ..models.response import Response
from ..models.retrieved_context import RetrievedContext
from ..services.qdrant_service import qdrant_service
from ..services.cohere_service import cohere_service
from ..services.postgres_service import postgres_service
from ..utils.text_splitter import TextSplitter


class RAGService:
    def __init__(self):
        self.qdrant_service = qdrant_service
        self.cohere_service = cohere_service
        self.postgres_service = postgres_service
        self.text_splitter = TextSplitter()

    async def process_query(self, query: Query) -> Response:
        """
        Process a user query and return a response with citations.
        """
        try:
            # Log the query
            await self.postgres_service.log_query(query)

            # Determine which content to search based on whether selected_text is provided
            search_content = query.selected_text if query.selected_text else "book"

            # Generate embedding for the query
            query_embeddings = await self.cohere_service.generate_embeddings([query.question])
            query_embedding = query_embeddings[0]

            # Search for relevant content
            if query.selected_text:
                # If user provided custom text, search in it first
                retrieved_contexts = await self._search_in_custom_text(query_embedding, query.selected_text)
            else:
                # Search in the book content
                retrieved_contexts = await self._search_in_book_content(query_embedding)

            # If no relevant content found in custom text, search in book content too
            if query.selected_text and not retrieved_contexts:
                book_contexts = await self._search_in_book_content(query_embedding)
                retrieved_contexts.extend(book_contexts)

            # If still no content found, return a response indicating so
            if not retrieved_contexts:
                response = {
                    "id": uuid4(),
                    "query_id": query.id,
                    "answer": "I couldn't find relevant information in the provided content to answer your question.",
                    "citations": [],
                    "confidence_score": 0.0,
                    "timestamp": datetime.utcnow(),
                    "was_answer_found": False
                }
            else:
                # Combine retrieved contexts into a prompt for the LLM
                context_str = "\n\n".join([ctx.content for ctx in retrieved_contexts])
                citations = list(set([ctx.book_section for ctx in retrieved_contexts if ctx.book_section]))

                # Determine the source of information for the prompt
                source_info = []
                for ctx in retrieved_contexts:
                    if ctx.source_type == "user_selected":
                        source_info.append("From user-provided text")
                    elif ctx.source_type == "book" and ctx.book_section:
                        source_info.append(f"From book section: {ctx.book_section}")
                    elif ctx.source_type == "book":
                        source_info.append("From book content")

                sources_str = "; ".join(set(source_info))  # Remove duplicates

                # Create a prompt for the LLM
                prompt = f"""
                Based on the following context, please answer the question.
                If the context doesn't contain enough information to answer the question,
                please say so clearly.

                Context: {context_str}

                Question: {query.question}

                Information sources: {sources_str}

                Please provide a concise, accurate answer and cite the relevant sections.
                """

                # Generate response using Cohere
                answer = await self.cohere_service.generate_response(prompt)

                response = {
                    "id": uuid4(),
                    "query_id": query.id,
                    "answer": answer,
                    "citations": citations,
                    "confidence_score": 0.8,  # Placeholder - would be calculated based on similarity scores
                    "timestamp": datetime.utcnow(),
                    "was_answer_found": True
                }

            # Log the response
            await self.postgres_service.log_response(response)

            # Create query history
            query_history = {
                "id": uuid4(),
                "query_id": query.id,
                "response_id": response["id"],
                "query_text": query.question,
                "response_text": response["answer"],
                "citations": response["citations"],
                "timestamp": response["timestamp"],
                "user_id": query.user_id,
                "session_id": query.session_id,
                "was_useful": None  # To be set later by user feedback
            }
            from ..models.query_history import QueryHistory
            query_history_model = QueryHistory(**query_history)
            await self.postgres_service.log_query_history(query_history_model)

            # Convert to Response model
            return Response(**response)

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            raise

    async def _search_in_custom_text(self, query_embedding: List[float], custom_text: str) -> List[RetrievedContext]:
        """
        Search for relevant content in the user-provided custom text.
        """
        try:
            # Split custom text into chunks
            text_chunks = self.text_splitter.split_text(custom_text)

            # For each chunk, we need to find similarity with the query
            # Since custom text isn't stored in Qdrant, we'll use Cohere's rerank functionality
            if not text_chunks:
                return []

            # Generate embeddings for the text chunks
            embeddings = await self.cohere_service.generate_embeddings(text_chunks)

            # Calculate similarity between query embedding and each text chunk embedding
            # For simplicity, we'll use a basic cosine similarity calculation
            similarities = []
            for emb in embeddings:
                similarity = self._cosine_similarity(query_embedding, emb)
                similarities.append(similarity)

            # Pair text chunks with their similarities and sort by similarity
            chunk_similarity_pairs = list(zip(text_chunks, similarities))
            chunk_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

            # Take top results
            top_pairs = chunk_similarity_pairs[:min(5, len(chunk_similarity_pairs))]

            # Convert to RetrievedContext models
            contexts = []
            for idx, (chunk, similarity) in enumerate(top_pairs):
                context = RetrievedContext(
                    id=uuid4(),
                    query_id=query.id,  # Use the actual query ID
                    content=chunk,
                    source_type="user_selected",
                    book_section=None,
                    similarity_score=similarity,
                    chunk_order=idx
                )
                contexts.append(context)

            return contexts
        except Exception as e:
            logging.error(f"Error searching in custom text: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))

            # Calculate magnitudes
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(a * a for a in vec2) ** 0.5

            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)
        except:
            # If there's an error calculating similarity, return 0
            return 0.0

    async def _search_in_book_content(self, query_embedding: List[float], limit: int = 5) -> List[RetrievedContext]:
        """
        Search for relevant content in the book content stored in Qdrant.
        """
        try:
            # Search in Qdrant for similar embeddings
            search_results = await self.qdrant_service.search_similar(
                query_vector=query_embedding,
                limit=limit
            )

            contexts = []
            for idx, result in enumerate(search_results):
                payload = result["payload"]
                context = RetrievedContext(
                    id=UUID(result["id"]),
                    query_id=uuid4(),  # This will be replaced with the actual query ID later
                    content=payload.get("content", ""),
                    source_type="book",
                    book_section=payload.get("section", payload.get("book_section", "")),
                    similarity_score=result["score"],
                    chunk_order=idx
                )
                contexts.append(context)

            return contexts
        except Exception as e:
            logging.error(f"Error searching in book content: {e}")
            return []

    async def ingest_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Ingest content into the system (store in Qdrant and potentially Postgres).
        """
        try:
            # Split content into chunks
            text_chunks = self.text_splitter.split_text(content)

            if not text_chunks:
                raise ValueError("No content to ingest after splitting")

            # Generate embeddings for all chunks
            embeddings = await self.cohere_service.generate_embeddings(text_chunks)

            # Store each chunk with its embedding
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                content_id = uuid4()

                # Prepare payload for Qdrant
                payload = {
                    "content": chunk,
                    "section_title": metadata.get("section_title", f"Section {i+1}"),
                    "page_reference": metadata.get("page_reference", f"Page {i+1}"),
                    "chapter": metadata.get("chapter", "Unknown Chapter"),
                    "source_type": "book"
                }

                # Store in Qdrant
                await self.qdrant_service.store_embedding(
                    content_id=content_id,
                    vector=embedding,
                    payload=payload
                )

            return f"Successfully ingested {len(text_chunks)} content chunks"
        except Exception as e:
            logging.error(f"Error ingesting content: {e}")
            raise


# Global instance
rag_service = RAGService()