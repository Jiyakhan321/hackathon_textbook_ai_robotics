from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import time
from ...models.query import Query
from ...services.rag_service import rag_service
from ...utils.validation import sanitize_input

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit a query to the RAG system and receive an AI-generated response with citations.
    """
    try:
        # Validate required fields
        if "question" not in query_data or not query_data["question"]:
            raise HTTPException(status_code=400, detail="Question is required and cannot be empty")

        # Sanitize and validate question
        question = query_data["question"]
        if not isinstance(question, str):
            raise HTTPException(status_code=400, detail="Question must be a string")

        # Sanitize the input
        question = sanitize_input(question)

        if len(question) < 1 or len(question) > 1000:
            raise HTTPException(status_code=400, detail="Question must be between 1 and 1000 characters")

        # Validate and sanitize selected_text if provided
        selected_text = query_data.get("selected_text")
        if selected_text is not None:
            if not isinstance(selected_text, str):
                raise HTTPException(status_code=400, detail="Selected text must be a string")

            # Sanitize the input
            selected_text = sanitize_input(selected_text)

            if len(selected_text) > 5000:
                raise HTTPException(status_code=400, detail="Selected text must be less than 5000 characters")

        # Validate user_id if provided
        user_id = query_data.get("user_id")
        if user_id is not None and not isinstance(user_id, str):
            raise HTTPException(status_code=400, detail="User ID must be a string")

        # Validate session_id if provided
        session_id = query_data.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="Session ID must be a string")

        # Create Query model instance
        query = Query(
            user_id=user_id,
            question=question,
            selected_text=selected_text,
            session_id=session_id
        )

        # Record start time for performance monitoring
        start_time = time.time()

        # Process the query using the RAG service
        response = await rag_service.process_query(query)

        # Calculate response time
        response_time = time.time() - start_time

        # Check if response time exceeds 2 seconds
        if response_time > 2.0:
            logging.warning(f"Response time exceeded 2 seconds: {response_time:.2f}s")

        # Format and return the response
        result = {
            "id": str(response.id),
            "query_id": str(response.query_id),
            "answer": response.answer,
            "citations": response.citations,
            "confidence_score": response.confidence_score,
            "was_answer_found": response.was_answer_found,
            "timestamp": response.timestamp.isoformat()
        }

        return result
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error occurred")