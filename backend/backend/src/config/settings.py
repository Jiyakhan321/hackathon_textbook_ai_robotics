from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    neon_db_url: str
    
    # Vector database settings
    qdrant_url: str
    qdrant_api_key: str
    
    # Cohere settings
    cohere_api_key: str
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Application settings
    app_name: str = "RAG Chatbot for AI Textbook"
    debug: bool = False
    version: str = "1.0.0"
    
    # RAG settings
    similarity_threshold: float = 0.7
    top_k_chunks: int = 5
    max_chunk_length: int = 1000  # tokens
    
    # Rate limiting
    requests_per_minute: int = 10
    
    # Cohere model settings
    cohere_model: str = "command-r-plus"
    cohere_embedding_model: str = "embed-multilingual-v3.0"

    class Config:
        env_file = ".env"


settings = Settings()