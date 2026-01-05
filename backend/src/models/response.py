from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID
from datetime import datetime


class Response(BaseModel):
    id: UUID
    query_id: UUID
    answer: str = Field(..., min_length=10, max_length=10000)
    citations: List[str] = Field(default_factory=list, max_items=10)
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    was_answer_found: bool

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "query_id": "123e4567-e89b-12d3-a456-426614174000",
                "answer": "The main concept of the book is...",
                "citations": ["Chapter 1, Section 1.1", "Page 45"],
                "confidence_score": 0.85,
                "timestamp": "2023-10-01T12:00:05Z",
                "was_answer_found": True
            }
        }