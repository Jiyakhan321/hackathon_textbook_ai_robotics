from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime


class Embedding(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    vector: List[float]
    model_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174002",
                "content_id": "123e4567-e89b-12d3-a456-426614174003",
                "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
                "model_name": "embed-english-v3.0",
                "created_at": "2023-10-01T12:00:00Z"
            }
        }