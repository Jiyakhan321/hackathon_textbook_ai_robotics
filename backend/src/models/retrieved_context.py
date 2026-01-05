from pydantic import BaseModel, Field
from typing import Literal, Optional
from uuid import UUID
from datetime import datetime


class RetrievedContext(BaseModel):
    id: UUID
    query_id: UUID
    content: str = Field(..., min_length=10, max_length=5000)
    source_type: Literal["book", "user_selected"]
    book_section: Optional[str] = None
    similarity_score: float = Field(ge=0.0, le=1.0)
    chunk_order: int

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174004",
                "query_id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "The main concept of the book is...",
                "source_type": "book",
                "book_section": "Chapter 1, Section 1.1",
                "similarity_score": 0.85,
                "chunk_order": 1
            }
        }