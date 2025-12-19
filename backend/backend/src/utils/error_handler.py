import logging
from fastapi import HTTPException, status
from typing import Dict, Any
import traceback
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)

logger = logging.getLogger(__name__)


class RAGException(HTTPException):
    """Custom exception for RAG-specific errors"""
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(status_code=status_code, detail=detail)


class BookContentNotFoundException(RAGException):
    """Raised when the book content cannot answer the user's question"""
    def __init__(self):
        super().__init__(
            detail="This information is not available in the book.",
            status_code=status.HTTP_404_NOT_FOUND
        )


class SelectedTextInsufficientException(RAGException):
    """Raised when the selected text is insufficient to answer the question"""
    def __init__(self):
        super().__init__(
            detail="The selected text does not contain enough information to answer this question.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


def log_query_attempt(query_text: str, context_mode: str, user_id: str = None):
    """Log a query attempt with relevant metadata (without the actual content)"""
    logger.info(f"Query attempt - Mode: {context_mode}, User: {user_id}")


def log_query_result(query_id: str, response_time_ms: int, has_sufficient_context: bool, chunk_ids: list = None):
    """Log the result of a query with metadata"""
    logger.info(f"Query result - ID: {query_id}, Time: {response_time_ms}ms, Sufficient context: {has_sufficient_context}")


def handle_error(error: Exception, context: str = ""):
    """Generic error handler with logging"""
    error_msg = f"Error in {context}: {str(error)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    return RAGException(detail=f"An error occurred: {str(error)}")