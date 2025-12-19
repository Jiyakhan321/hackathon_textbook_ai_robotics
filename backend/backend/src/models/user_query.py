from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from ..config.database import Base


class ContextMode(str, Enum):
    full_book = "full_book"
    selected_text = "selected_text"


# SQLAlchemy Model for UserQuery
class UserQuerySQL(Base):
    __tablename__ = "user_queries"

    query_id = Column(String, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    context_mode = Column(String, nullable=False)  # full_book or selected_text
    selected_text = Column(Text, nullable=True)  # Only for selected_text mode
    user_id = Column(String, nullable=True, index=True)
    session_id = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


# Pydantic Model for UserQuery
class UserQueryBase(BaseModel):
    query_id: str
    query_text: str
    context_mode: ContextMode
    selected_text: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True


class UserQueryCreate(BaseModel):
    query_text: str
    context_mode: ContextMode
    selected_text: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    class Config:
        from_attributes = True


class UserQueryUpdate(BaseModel):
    query_text: Optional[str] = None
    context_mode: Optional[ContextMode] = None
    selected_text: Optional[str] = None

    class Config:
        from_attributes = True