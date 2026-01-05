from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime


class Query(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: Optional[str] = None
    question: str = Field(..., min_length=1, max_length=1000)
    selected_text: Optional[str] = Field(None, max_length=5000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user123",
                "question": "What is the main concept of this book?",
                "selected_text": "Optional custom text provided by the user",
                "timestamp": "2023-10-01T12:00:00Z",
                "session_id": "session456"
            }
        }