import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from src.api.main import app
from src.models.response import Response as ModelResponse
from src.services.rag_service import RAGService
from uuid import UUID


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint_success(client):
    """Test successful chat endpoint request."""
    # Mock the RAG service response
    mock_response = ModelResponse(
        id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        query_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        answer="The main concept of the book is artificial intelligence.",
        citations=["Chapter 1", "Section 2.3"],
        confidence_score=0.85,
        was_answer_found=True
    )

    # Patch the RAG service to return our mock response
    with patch('src.api.endpoints.chat.rag_service') as mock_rag_service:
        mock_rag_service.process_query = AsyncMock(return_value=mock_response)

        response = client.post(
            "/chat",
            json={
                "question": "What is the main concept of this book?",
                "user_id": "test_user",
                "session_id": "test_session"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The main concept of the book is artificial intelligence."
        assert "Chapter 1" in data["citations"]
        assert data["confidence_score"] == 0.85
        assert data["was_answer_found"] is True


@pytest.mark.asyncio
async def test_chat_endpoint_with_selected_text(client):
    """Test chat endpoint with selected text."""
    mock_response = ModelResponse(
        id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        query_id=UUID("123e4567-e89b-12d3-a456-426614174001"),
        answer="Custom text provides additional context.",
        citations=["User provided text"],
        confidence_score=0.9,
        was_answer_found=True
    )

    with patch('src.api.endpoints.chat.rag_service') as mock_rag_service:
        mock_rag_service.process_query = AsyncMock(return_value=mock_response)

        response = client.post(
            "/chat",
            json={
                "question": "Based on the selected text, what does it say?",
                "selected_text": "This is the selected text that provides additional context.",
                "user_id": "test_user"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "custom text" in data["answer"].lower()


def test_chat_endpoint_missing_question(client):
    """Test chat endpoint with missing question."""
    response = client.post(
        "/chat",
        json={
            "user_id": "test_user"
        }
    )

    assert response.status_code == 400
    assert "Question is required" in response.json()["detail"]


def test_chat_endpoint_empty_question(client):
    """Test chat endpoint with empty question."""
    response = client.post(
        "/chat",
        json={
            "question": "",
            "user_id": "test_user"
        }
    )

    assert response.status_code == 400
    assert "Question is required" in response.json()["detail"]


def test_chat_endpoint_too_long_question(client):
    """Test chat endpoint with too long question."""
    response = client.post(
        "/chat",
        json={
            "question": "a" * 1001,  # More than 1000 characters
            "user_id": "test_user"
        }
    )

    assert response.status_code == 400
    assert "between 1 and 1000 characters" in response.json()["detail"]


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "dependencies" in data