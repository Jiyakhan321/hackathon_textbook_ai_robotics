from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime


class QueryHistory(BaseModel):
    id: UUID
    query_id: UUID
    response_id: UUID
    query_text: str = Field(..., min_length=1, max_length=1000)
    response_text: str = Field(..., min_length=10, max_length=10000)
    citations: Optional[list[str]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    was_useful: Optional[bool] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174003",
                "query_id": "123e4567-e89b-12d3-a456-426614174000",
                "response_id": "123e4567-e89b-12d3-a456-426614174001",
                "query_text": "What is the main concept of this book?",
                "response_text": "The main concept of the book is...",
                "citations": ["Chapter 1, Section 1.1", "Page 45"],
                "timestamp": "2023-10-01T12:00:05Z",
                "user_id": "user123",
                "session_id": "session456",
                "was_useful": True
            }
        }