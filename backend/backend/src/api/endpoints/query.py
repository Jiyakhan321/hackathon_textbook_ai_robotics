from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from ...config.database import get_db
from ...models.user_query import UserQueryCreate, ContextMode
from ...models.response import ResponseCreate
from ...services.rag_service import RAGService
from ...utils.error_handler import RAGException, BookContentNotFoundException, handle_error
from pydantic import BaseModel, Field


router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to be answered")
    context_mode: str = Field(default="full_book", description="The context mode (fixed for this endpoint)")


class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    context_mode: str
    timestamp: str


@router.post("/query", response_model=QueryResponse)
def query_full_book(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    try:
        # Validate input
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Create a UserQueryCreate object
        user_query = UserQueryCreate(
            query_text=request.question,
            context_mode=ContextMode.full_book
        )

        # Initialize RAG service and process the query
        rag_service = RAGService()
        response = rag_service.process_query(db, user_query)

        # Format the sources for the response
        sources = []  # In a full implementation, this would be populated from the chunk metadata

        # Prepare the response in the required format
        api_response = QueryResponse(
            answer=response.answer_text,
            sources=sources,
            confidence=response.confidence_score,
            context_mode=response.context_mode,
            timestamp="2025-12-18T10:30:00Z"  # In real implementation, would be actual timestamp
        )

        return api_response

    except BookContentNotFoundException:
        # Return the specific response when content is not available in the book
        raise HTTPException(
            status_code=404,
            detail="This information is not available in the book."
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except RAGException as e:
        # Handle custom RAG exceptions
        raise e
    except Exception as e:
        # Handle other exceptions
        error = handle_error(e, "query endpoint")
        raise error