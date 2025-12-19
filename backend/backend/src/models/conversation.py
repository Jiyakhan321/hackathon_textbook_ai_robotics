from sqlalchemy import Column, String, DateTime, Boolean
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from ..config.database import Base


# SQLAlchemy Model for Conversation
class ConversationSQL(Base):
    __tablename__ = "conversations"

    conversation_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)


# Pydantic Model for Conversation
class ConversationBase(BaseModel):
    conversation_id: str
    user_id: Optional[str] = None
    session_id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    user_id: Optional[str] = None
    session_id: str
    is_active: Optional[bool] = True

    class Config:
        from_attributes = True


class ConversationUpdate(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    is_active: Optional[bool] = None

    class Config:
        from_attributes = True