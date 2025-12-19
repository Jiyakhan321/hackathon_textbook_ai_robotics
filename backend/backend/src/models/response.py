from sqlalchemy import Column, String, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from ..config.database import Base
from .user_query import ContextMode


# SQLAlchemy Model for Response
class ResponseSQL(Base):
    __tablename__ = "responses"

    response_id = Column(String, primary_key=True, index=True)
    answer_text = Column(Text, nullable=False)
    source_chunks = Column(ARRAY(String), nullable=False)  # Array of chunk_ids
    confidence_score = Column(Float, nullable=False)
    query_id = Column(String, nullable=False, index=True)  # Reference to the original query
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    context_mode = Column(String, nullable=False)  # To track which mode was used


# Pydantic Model for Response
class ResponseBase(BaseModel):
    response_id: str
    answer_text: str
    source_chunks: List[str]
    confidence_score: float
    query_id: str
    timestamp: datetime
    context_mode: ContextMode

    class Config:
        from_attributes = True


class ResponseCreate(BaseModel):
    answer_text: str
    source_chunks: List[str]
    confidence_score: float
    query_id: str
    context_mode: ContextMode

    class Config:
        from_attributes = True


class ResponseUpdate(BaseModel):
    answer_text: Optional[str] = None
    source_chunks: Optional[List[str]] = None
    confidence_score: Optional[float] = None

    class Config:
        from_attributes = True