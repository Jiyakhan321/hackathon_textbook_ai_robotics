from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from dotenv import load_dotenv

# Import modules with proper path handling for Vercel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.query import query_rag, query_selected_text
from models.schemas import QueryRequest, QueryResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG-based chatbot that answers questions from book content",
    version="1.0.0"
)

# Add CORS middleware to allow requests from Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hackathon-textbook-ai-robotics.vercel.app", "http://localhost:3000", "http://localhost:3001"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/chat", response_model=QueryResponse)
async def chat_general(request: QueryRequest):
    """Answer questions based on the entire book content"""
    try:
        logger.info(f"Processing general query: {request.query[:50]}...")
        response = await query_rag(request.query, request.context)
        return response
    except Exception as e:
        logger.error(f"General chat query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/chat/selected", response_model=QueryResponse)
async def chat_selected_text(request: QueryRequest):
    """Answer questions based only on selected text"""
    try:
        logger.info(f"Processing selected text query: {request.query[:50]}...")
        if not request.selected_text:
            raise HTTPException(status_code=400, detail="Selected text is required for this endpoint")
        response = await query_selected_text(request.query, request.selected_text)
        return response
    except Exception as e:
        logger.error(f"Selected text query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# For Vercel compatibility
from mangum import Mangum
handler = Mangum(app)

# For local development compatibility
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "index:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )