from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
from ...models.book_content import BookContent
from ...services.rag_service import rag_service

router = APIRouter()


@router.post("/ingest")
async def ingest_content(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest book content or user-provided text into the vector database.
    """
    try:
        # Validate required fields
        if "content" not in content_data:
            raise HTTPException(status_code=400, detail="Content is required")

        content = content_data["content"]
        metadata = content_data.get("metadata", {})

        # Validate content length
        if len(content) < 50:
            raise HTTPException(status_code=400, detail="Content must be at least 50 characters long")

        if len(content) > 10000:
            raise HTTPException(status_code=400, detail="Content must be less than 10000 characters")

        # Validate source type if provided
        source_type = content_data.get("source_type", "book")
        if source_type not in ["book", "user_content"]:
            raise HTTPException(status_code=400, detail="source_type must be 'book' or 'user_content'")

        # Ingest the content using the RAG service
        result = await rag_service.ingest_content(content, metadata)

        return {
            "status": "success",
            "message": result,
            "vector_id": "N/A"  # The actual vector ID isn't returned by the current implementation
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error in ingestion endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")