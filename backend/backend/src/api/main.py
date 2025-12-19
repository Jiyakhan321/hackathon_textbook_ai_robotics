from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.config.settings import settings
from src.config.vector_db import initialize_qdrant_collection
from src.api.endpoints import query, query_selected_text, widget
from src.middleware.auth_middleware import api_key_auth


# Create FastAPI app with settings
app = FastAPI(
    title=settings.app_name,
    description="API for the RAG Chatbot for AI Textbook",
    version=settings.version,
    debug=settings.debug
)

# Add CORS middleware to allow communication with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(query_selected_text.router, prefix="/api", tags=["query-selected-text"])
app.include_router(widget.router, prefix="/api", tags=["widget"])

# Root endpoint
@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.app_name} v{settings.version}"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": settings.version}

# Initialize database connections and vector DB on startup
@app.on_event("startup")
async def startup_event():
    # Initialize Qdrant collection
    initialize_qdrant_collection()
    print("Application startup complete")