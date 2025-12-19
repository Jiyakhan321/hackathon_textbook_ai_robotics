import logging
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from ..models.query_log import QueryLog  # We'll create this model soon


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_query_metadata(
    db: Session,
    query_id: str,
    response_id: str,
    chunk_ids: List[str],
    similarity_scores: List[float],
    response_time_ms: int,
    has_sufficient_context: bool
):
    """
    Log query metadata for analytics and debugging.
    This complies with the constitutional requirement to NOT log raw book text.
    """
    try:
        # Create a QueryLog entry with metadata only (no actual content)
        query_log = QueryLog(
            log_id=f"log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}",
            query_id=query_id,
            response_id=response_id,
            chunk_ids=chunk_ids,
            similarity_scores=similarity_scores,
            response_time_ms=response_time_ms,
            has_sufficient_context=has_sufficient_context,
            timestamp=datetime.utcnow()
        )
        
        # Add to database
        db.add(query_log)
        db.commit()
        
        logger.info(f"Logged query metadata for query {query_id}")
    except Exception as e:
        logger.error(f"Failed to log query metadata: {str(e)}")
        # Don't raise exception as logging shouldn't break the main flow