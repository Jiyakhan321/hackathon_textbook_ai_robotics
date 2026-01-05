from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime


class BookContent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    section_title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=50, max_length=2000)
    page_reference: str
    chapter: str
    vector_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174005",
                "section_title": "Introduction to AI",
                "content": "Artificial intelligence is a branch of computer science...",
                "page_reference": "Page 15",
                "chapter": "Chapter 1",
                "vector_id": "vector_12345",
                "created_at": "2023-10-01T12:00:00Z"
            }
        }