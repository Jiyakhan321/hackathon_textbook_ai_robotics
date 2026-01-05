import pytest
from uuid import UUID, uuid4
from datetime import datetime
from src.models.response import Response


def test_response_creation():
    """Test creating a Response model with valid data."""
    response_id = uuid4()
    query_id = uuid4()

    response = Response(
        id=response_id,
        query_id=query_id,
        answer="The main concept of the book is artificial intelligence.",
        citations=["Chapter 1", "Section 2.3"],
        confidence_score=0.85,
        was_answer_found=True
    )

    assert response.id == response_id
    assert response.query_id == query_id
    assert response.answer == "The main concept of the book is artificial intelligence."
    assert response.citations == ["Chapter 1", "Section 2.3"]
    assert response.confidence_score == 0.85
    assert response.was_answer_found is True
    assert isinstance(response.timestamp, datetime)


def test_response_required_fields():
    """Test that required fields are validated."""
    # Missing required fields should fail
    with pytest.raises(ValueError):
        Response(
            id=uuid4(),
            query_id=uuid4(),
            # answer is missing
            citations=["Chapter 1"],
            confidence_score=0.85,
            was_answer_found=True
        )


def test_response_answer_length():
    """Test answer length validation."""
    # Too short answer should fail
    with pytest.raises(ValueError):
        Response(
            id=uuid4(),
            query_id=uuid4(),
            answer="",  # Too short
            citations=["Chapter 1"],
            confidence_score=0.85,
            was_answer_found=True
        )

    # Too long answer should fail
    with pytest.raises(ValueError):
        Response(
            id=uuid4(),
            query_id=uuid4(),
            answer="a" * 10001,  # Too long
            citations=["Chapter 1"],
            confidence_score=0.85,
            was_answer_found=True
        )

    # Valid length should pass
    response = Response(
        id=uuid4(),
        query_id=uuid4(),
        answer="a" * 10,  # Minimum length
        citations=["Chapter 1"],
        confidence_score=0.85,
        was_answer_found=True
    )
    assert len(response.answer) == 10


def test_response_confidence_score_validation():
    """Test confidence score validation."""
    # Score too low should fail
    with pytest.raises(ValueError):
        Response(
            id=uuid4(),
            query_id=uuid4(),
            answer="Test answer",
            citations=["Chapter 1"],
            confidence_score=-0.1,  # Below 0.0
            was_answer_found=True
        )

    # Score too high should fail
    with pytest.raises(ValueError):
        Response(
            id=uuid4(),
            query_id=uuid4(),
            answer="Test answer",
            citations=["Chapter 1"],
            confidence_score=1.1,  # Above 1.0
            was_answer_found=True
        )

    # Valid scores should pass
    response = Response(
        id=uuid4(),
        query_id=uuid4(),
        answer="Test answer",
        citations=["Chapter 1"],
        confidence_score=0.5,  # Valid score
        was_answer_found=True
    )
    assert response.confidence_score == 0.5


def test_response_citations_validation():
    """Test citations list validation."""
    # Too many citations should fail
    with pytest.raises(ValueError):
        Response(
            id=uuid4(),
            query_id=uuid4(),
            answer="Test answer",
            citations=[f"citation_{i}" for i in range(11)],  # More than 10
            confidence_score=0.5,
            was_answer_found=True
        )

    # Valid number of citations should pass
    response = Response(
        id=uuid4(),
        query_id=uuid4(),
        answer="Test answer",
        citations=[f"citation_{i}" for i in range(10)],  # Exactly 10
        confidence_score=0.5,
        was_answer_found=True
    )
    assert len(response.citations) == 10