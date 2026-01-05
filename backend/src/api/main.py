from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

load_dotenv()

from ..services.qdrant_service import qdrant_service
from ..services.postgres_service import postgres_service
from ..services.cohere_service import cohere_service
from ..config.settings import settings
from .middleware import RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    print("Starting up RAG Chatbot API...")

    await qdrant_service.initialize()
    await postgres_service.initialize()
    await cohere_service.initialize()

    yield
    print("Shutting down RAG Chatbot API...")


app = FastAPI(
    title="RAG Chatbot for Published AI Book",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(",")
    if settings.allowed_origins != "*"
    else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .endpoints.health import router as health_router
from .endpoints.chat import router as chat_router
from .endpoints.ingestion import router as ingestion_router

app.include_router(chat_router)
app.include_router(ingestion_router)
app.include_router(health_router)
