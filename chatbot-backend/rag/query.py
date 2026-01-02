import asyncio
from typing import Optional, List
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient
import logging
import os
from dotenv import load_dotenv

from models.schemas import QueryResponse
from db.qdrant import get_qdrant_client
from rag.embeddings import embed_text

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize OpenRouter model (lazy initialization to avoid import errors)
def get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Either OPENROUTER_API_KEY or OPENAI_API_KEY must be set in environment variables")

    return ChatOpenRouter(
        model="openai/gpt-4o",  # Using GPT-4o as a high-quality model available on OpenRouter
        temperature=0.1,
        openrouter_api_key=api_key
    )

async def query_rag(query: str, context: Optional[str] = None) -> QueryResponse:
    """
    Query the RAG system with the entire book content
    """
    try:
        # Initialize Qdrant client
        qdrant_client = get_qdrant_client()

        # Create embedding for the query using our custom embedding function
        query_embedding = embed_text(query)

        # Search in Qdrant for relevant chunks
        search_result = qdrant_client.search(
            collection_name="book_chunks",
            query_vector=query_embedding,
            limit=5,  # Retrieve top 5 most relevant chunks
            with_payload=True
        )

        # Extract relevant content
        relevant_chunks = []
        sources = []
        for hit in search_result:
            chunk_content = hit.payload.get("content", "")
            source_info = {
                "title": hit.payload.get("title", "Unknown"),
                "source": hit.payload.get("source", "Unknown")
            }
            relevant_chunks.append(chunk_content)
            sources.append(source_info)

        if not relevant_chunks:
            response = QueryResponse(
                response="I couldn't find any information about this in the book. Please check if your question is related to the book content.",
                sources=[],
                query=query
            )
            return response

        # Combine context and relevant chunks
        context_str = "\n\n".join(relevant_chunks)
        if context:
            context_str = f"{context}\n\n{context_str}"

        # Create the prompt for the LLM
        system_message = SystemMessage(
            content="You are an AI assistant that answers questions based on provided book content. "
                    "Only use the information provided in the context to answer the question. "
                    "If the answer is not in the context, clearly state that the information is not in the book. "
                    "Be concise and accurate in your responses."
        )

        human_message = HumanMessage(
            content=f"Context: {context_str}\n\nQuestion: {query}\n\nAnswer:"
        )

        # Get response from LLM
        llm = get_llm()
        result = llm.invoke([system_message, human_message])
        answer = result.content

        response = QueryResponse(
            response=answer,
            sources=sources,
            query=query
        )

        return response

    except Exception as e:
        logger.error(f"Error during RAG query: {str(e)}")
        raise e

async def query_selected_text(query: str, selected_text: str) -> QueryResponse:
    """
    Query the RAG system with only the selected text
    """
    try:
        if not selected_text or not selected_text.strip():
            response = QueryResponse(
                response="No selected text provided. Please select some text from the book to ask about.",
                sources=[{"title": "Selected Text", "source": "User Selection"}],
                query=query
            )
            return response

        # Create the prompt for the LLM using only the selected text
        system_message = SystemMessage(
            content="You are an AI assistant that answers questions based only on the provided selected text. "
                    "Do not use any external knowledge. Only answer based on the provided text. "
                    "If the answer is not in the selected text, clearly state that the information is not in the selected text. "
                    "Be concise and accurate in your responses."
        )

        human_message = HumanMessage(
            content=f"Selected text: {selected_text}\n\nQuestion: {query}\n\nAnswer:"
        )

        # Get response from LLM
        llm = get_llm()
        result = llm.invoke([system_message, human_message])
        answer = result.content

        response = QueryResponse(
            response=answer,
            sources=[{"title": "Selected Text", "source": "User Selection"}],
            query=query
        )

        return response

    except Exception as e:
        logger.error(f"Error during selected text query: {str(e)}")
        raise e