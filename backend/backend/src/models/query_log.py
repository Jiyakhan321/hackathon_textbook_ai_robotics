from sqlalchemy import Column, String, Integer, DateTime, Boolean
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from ..config.database import Base


# SQLAlchemy Model for QueryLog
class QueryLogSQL(Base):
    __tablename__ = "query_logs"

    log_id = Column(String, primary_key=True, index=True)
    query_id = Column(String, nullable=False, index=True)  # Reference to the original query
    response_id = Column(String, nullable=False, index=True)  # Reference to the generated response
    chunk_ids = Column(ARRAY(String), nullable=False)  # IDs of all retrieved chunks
    similarity_scores = Column(ARRAY(Integer), nullable=False)  # Similarity scores for retrieved chunks (multiplied by 100 to store as int)
    response_time_ms = Column(Integer, nullable=False)  # Time taken to generate response in milliseconds
    has_sufficient_context = Column(Boolean, nullable=False)  # Whether sufficient context was available
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)  # When the log entry was created


# Pydantic Model for QueryLog
class QueryLogBase(BaseModel):
    log_id: str
    query_id: str
    response_id: str
    chunk_ids: List[str]
    similarity_scores: List[int]  # Stored as integer percentages (original float * 100)
    response_time_ms: int
    has_sufficient_context: bool
    timestamp: datetime

    class Config:
        from_attributes = True


class QueryLogCreate(BaseModel):
    query_id: str
    response_id: str
    chunk_ids: List[str]
    similarity_scores: List[int]  # Stored as integer percentages
    response_time_ms: int
    has_sufficient_context: bool

    class Config:
        from_attributes = True