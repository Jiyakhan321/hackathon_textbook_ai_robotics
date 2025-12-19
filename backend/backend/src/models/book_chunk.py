from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.mutable import MutableDict
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
from ..config.database import Base


# SQLAlchemy Model for BookChunk
class BookChunkSQL(Base):
    __tablename__ = "book_chunks"

    chunk_id = Column(String, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    chapter_name = Column(String, nullable=False, index=True)
    section_title = Column(String, nullable=False, index=True)
    page_number = Column(Integer, nullable=True)
    source_reference = Column(String, nullable=False)
    # For embedding vectors, we'll store as JSON array since SQLAlchemy doesn't have native vector type
    embedding_vector = Column(String, nullable=False)  # Stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True)

    def set_embedding_vector(self, vector: List[float]):
        self.embedding_vector = json.dumps(vector)

    def get_embedding_vector(self) -> List[float]:
        return json.loads(self.embedding_vector)


# Pydantic Model for BookChunk
class BookChunkBase(BaseModel):
    chunk_id: str
    content: str
    chapter_name: str
    section_title: str
    page_number: Optional[int] = None
    source_reference: str
    embedding_vector: List[float]  # This will be handled differently in the actual implementation
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BookChunkCreate(BaseModel):
    content: str
    chapter_name: str
    section_title: str
    page_number: Optional[int] = None
    source_reference: str
    embedding_vector: Optional[List[float]] = None  # Will be computed during creation

    class Config:
        from_attributes = True


class BookChunkUpdate(BaseModel):
    content: Optional[str] = None
    chapter_name: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    source_reference: Optional[str] = None

    class Config:
        from_attributes = True