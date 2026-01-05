from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    neon_url: str

    # Qdrant settings
    qdrant_link: str
    qdrant_api_key: str

    # Cohere settings
    cohere_api_key: str

    # Application settings
    app_name: str = "RAG Chatbot for Published AI Book"
    debug: bool = False
    allowed_origins: str = "*"  # Should be configured properly in production

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()