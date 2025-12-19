from typing import List, Tuple
from ..models.user_query import UserQueryCreate, ContextMode
from ..models.response import ResponseCreate
from ..models.book_chunk import BookChunkCreate, BookChunkBase
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import RetrievalService
from ..services.cohere_service import CohereService
from ..utils.error_handler import BookContentNotFoundException, SelectedTextInsufficientException
from ..utils.query_logger import log_query_metadata
from sqlalchemy.orm import Session
from datetime import datetime
import uuid


class RAGService:
    """
    Main service that orchestrates the RAG (Retrieval-Augmented Generation) flow
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.retrieval_service = RetrievalService()
        self.cohere_service = CohereService()

    def process_query(self, db: Session, user_query: UserQueryCreate) -> ResponseCreate:
        """
        Process a user query through the full RAG pipeline
        """
        # Generate a unique query ID
        query_id = f"query_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Log the query attempt
        from ..utils.error_handler import log_query_attempt
        log_query_attempt(user_query.query_text, user_query.context_mode, user_query.user_id)

        start_time = datetime.utcnow()

        try:
            # Step 1: Generate embedding for the query
            query_embedding = self.embedding_service.embed_query(user_query.query_text)

            # Step 2: Retrieve relevant chunks based on the query embedding
            if user_query.context_mode == ContextMode.selected_text and user_query.selected_text:
                # For selected text mode, we need to temporarily handle the selected text
                # In this case, we'll create temporary chunks from the selected text
                from ..services.chunking_service import ChunkingService
                temp_chunks = ChunkingService.chunk_selected_text(user_query.selected_text)

                # Embed these temporary chunks
                temp_chunks_with_embeddings = self.embedding_service.embed_book_chunks(temp_chunks)

                # Since these are temporary chunks from selected text, we'll treat them as relevant
                relevant_chunks_with_scores = []
                for chunk in temp_chunks_with_embeddings:
                    # For temporary chunks, we'll use a high similarity score since it's the exact selected text
                    relevant_chunks_with_scores.append((chunk, 1.0))
            else:
                # For full book mode, retrieve from the vector database
                relevant_chunks_with_scores = self.retrieval_service.retrieve_relevant_chunks(
                    query_embedding,
                    context_mode=user_query.context_mode
                )

            # Extract just the chunks (without scores for now)
            relevant_chunks = [chunk_score[0] for chunk_score in relevant_chunks_with_scores]

            # Check if we found relevant chunks
            if not relevant_chunks:
                if user_query.context_mode == ContextMode.selected_text:
                    # For selected text mode, if no chunks found, it means the text wasn't sufficient
                    raise SelectedTextInsufficientException()
                else:
                    # For full book mode, if no chunks found, content is not available
                    raise BookContentNotFoundException()

            # Check if the context is sufficient to answer the question
            is_context_sufficient = self.cohere_service.check_context_sufficiency(
                user_query.query_text,
                relevant_chunks
            )

            if not is_context_sufficient:
                if user_query.context_mode == ContextMode.selected_text:
                    raise SelectedTextInsufficientException()
                else:
                    raise BookContentNotFoundException()

            # Step 3: Generate response using the relevant chunks
            response_text = self.cohere_service.generate_response(
                user_query.query_text,
                relevant_chunks,
                user_query.context_mode
            )

            # Step 4: Create response object
            response_id = f"resp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Get chunk IDs for the source attribution
            source_chunk_ids = [chunk.chunk_id for chunk in relevant_chunks]

            # Calculate confidence based on similarity scores (average of top scores)
            confidence_score = sum([score for _, score in relevant_chunks_with_scores]) / len(relevant_chunks_with_scores) if relevant_chunks_with_scores else 0.0

            # Create the response object
            response = ResponseCreate(
                answer_text=response_text,
                source_chunks=source_chunk_ids,
                confidence_score=confidence_score,
                query_id=query_id,
                context_mode=user_query.context_mode
            )

            # Log the query result
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Extract similarity scores for logging
            similarity_scores = [int(score * 100) for _, score in relevant_chunks_with_scores]  # Convert to percentage

            from ..utils.error_handler import log_query_result
            log_query_result(query_id, response_time_ms, True, source_chunk_ids)

            # Log query metadata (without raw content)
            log_query_metadata(
                db,
                query_id,
                response_id,
                source_chunk_ids,
                similarity_scores,
                response_time_ms,
                True
            )

            return response

        except BookContentNotFoundException:
            # Log the failed query result
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            from ..utils.error_handler import log_query_result
            log_query_result(query_id, response_time_ms, False, [])

            # Log query metadata for failed query (without raw content)
            log_query_metadata(
                db,
                query_id,
                "none",
                [],
                [],
                response_time_ms,
                False
            )

            raise
        except SelectedTextInsufficientException:
            # Log the failed query result
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            from ..utils.error_handler import log_query_result
            log_query_result(query_id, response_time_ms, False, [])

            # Log query metadata for failed query (without raw content)
            log_query_metadata(
                db,
                query_id,
                "none",
                [],
                [],
                response_time_ms,
                False
            )

            raise
        except Exception as e:
            # Log the error
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            from ..utils.error_handler import logger
            logger.error(f"Error processing query {query_id}: {str(e)}")

            # Log query metadata for failed query (without raw content)
            log_query_metadata(
                db,
                query_id,
                "none",
                [],
                [],
                response_time_ms,
                False
            )

            # Re-raise the exception to be handled by the calling layer
            raise

    def process_selected_text_query(self, db: Session, user_query: UserQueryCreate) -> ResponseCreate:
        """
        Process a query specifically using selected text (ensures it ignores global book index)
        """
        # This function specifically handles selected text queries
        # It ensures that the global book corpus is completely ignored
        # and only uses the provided selected_text

        # Validate that selected text is provided
        if not user_query.selected_text:
            raise ValueError("Selected text is required for selected text mode")

        # Generate a unique query ID
        query_id = f"query_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Log the query attempt
        from ..utils.error_handler import log_query_attempt
        log_query_attempt(user_query.query_text, user_query.context_mode, user_query.user_id)

        start_time = datetime.utcnow()

        try:
            # Create temporary chunks from the selected text
            from ..services.chunking_service import ChunkingService
            temp_chunks = ChunkingService.chunk_selected_text(user_query.selected_text)

            # Embed these temporary chunks
            temp_chunks_with_embeddings = self.embedding_service.embed_book_chunks(temp_chunks)

            # Treat these as the relevant chunks for the query
            relevant_chunks_with_scores = [(chunk, 1.0) for chunk in temp_chunks_with_embeddings]
            relevant_chunks = [chunk_score[0] for chunk_score in relevant_chunks_with_scores]

            # Check if the context is sufficient to answer the question
            is_context_sufficient = self.cohere_service.check_context_sufficiency(
                user_query.query_text,
                relevant_chunks
            )

            if not is_context_sufficient:
                raise SelectedTextInsufficientException()

            # Generate response using only the selected text chunks
            response_text = self.cohere_service.generate_response(
                user_query.query_text,
                relevant_chunks,
                context_mode="selected_text"
            )

            # Create response object
            response_id = f"resp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Get chunk IDs for the source attribution
            source_chunk_ids = [chunk.chunk_id for chunk in relevant_chunks]

            # For selected text mode, we'll set the confidence based on the sufficiency check
            confidence_score = 0.8 if is_context_sufficient else 0.3

            # Create the response object
            response = ResponseCreate(
                answer_text=response_text,
                source_chunks=source_chunk_ids,
                confidence_score=confidence_score,
                query_id=query_id,
                context_mode=ContextMode.selected_text
            )

            # Log the query result
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Extract similarity scores for logging
            similarity_scores = [int(score * 100) for _, score in relevant_chunks_with_scores]  # Convert to percentage

            from ..utils.error_handler import log_query_result
            log_query_result(query_id, response_time_ms, True, source_chunk_ids)

            # Log query metadata (without raw content)
            log_query_metadata(
                db,
                query_id,
                response_id,
                source_chunk_ids,
                similarity_scores,
                response_time_ms,
                True
            )

            return response

        except SelectedTextInsufficientException:
            # Log the failed query result
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            from ..utils.error_handler import log_query_result
            log_query_result(query_id, response_time_ms, False, [])

            # Log query metadata for failed query (without raw content)
            log_query_metadata(
                db,
                query_id,
                "none",
                [],
                [],
                response_time_ms,
                False
            )

            raise
        except Exception as e:
            # Log the error
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            from ..utils.error_handler import logger
            logger.error(f"Error processing selected text query {query_id}: {str(e)}")

            # Log query metadata for failed query (without raw content)
            log_query_metadata(
                db,
                query_id,
                "none",
                [],
                [],
                response_time_ms,
                False
            )

            # Re-raise the exception to be handled by the calling layer
            raise