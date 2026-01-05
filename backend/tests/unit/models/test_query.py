import pytest
from uuid import UUID, uuid4
from datetime import datetime
from src.models.query import Query


def test_query_creation():
    """Test creating a Query model with valid data."""
    query = Query(
        user_id="test_user",
        question="What is the main concept of this book?",
        selected_text="Some custom text",
        session_id="test_session"
    )

    assert query.user_id == "test_user"
    assert query.question == "What is the main concept of this book?"
    assert query.selected_text == "Some custom text"
    assert query.session_id == "test_session"
    assert isinstance(query.id, UUID)
    assert isinstance(query.timestamp, datetime)


def test_query_required_fields():
    """Test that question is required."""
    with pytest.raises(ValueError):
        Query(
            user_id="test_user",
            question="",  # Empty question should fail
            selected_text="Some custom text"
        )


def test_query_question_length():
    """Test question length validation."""
    # Too long question should fail
    with pytest.raises(ValueError):
        Query(
            user_id="test_user",
            question="a" * 1001,  # More than 1000 characters
            selected_text="Some custom text"
        )

    # Valid length should pass
    query = Query(
        user_id="test_user",
        question="a" * 1000,  # Exactly 1000 characters
        selected_text="Some custom text"
    )
    assert len(query.question) == 1000


def test_query_selected_text_length():
    """Test selected text length validation."""
    # Too long selected text should fail
    with pytest.raises(ValueError):
        Query(
            user_id="test_user",
            question="What is this?",
            selected_text="a" * 5001  # More than 5000 characters
        )

    # Valid length should pass
    query = Query(
        user_id="test_user",
        question="What is this?",
        selected_text="a" * 5000  # Exactly 5000 characters
    )
    assert len(query.selected_text) == 5000


def test_query_optional_fields():
    """Test that optional fields can be None."""
    query = Query(
        question="What is the main concept of this book?"
        # user_id, selected_text, and session_id are all optional
    )

    assert query.user_id is None
    assert query.selected_text is None
    assert query.session_id is None
    assert query.question == "What is the main concept of this book?"