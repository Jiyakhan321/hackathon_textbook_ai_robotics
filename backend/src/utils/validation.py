import re
from typing import Optional
from ..models.query import Query


def validate_query_input(query: str) -> bool:
    """
    Validate query input to ensure it meets requirements.
    """
    if not query or len(query.strip()) == 0:
        return False
    if len(query) > 1000:
        return False
    return True


def validate_selected_text_input(selected_text: Optional[str]) -> bool:
    """
    Validate selected text input to ensure it meets requirements.
    """
    if selected_text is None:
        return True
    if len(selected_text) > 5000:
        return False
    return True


def sanitize_input(text: str) -> str:
    """
    Sanitize input to prevent injection attacks.
    """
    if not text:
        return text

    # Remove potentially dangerous characters/sequences
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)

    # Remove other potentially harmful patterns
    sanitized = re.sub(r'eval\(', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'expression\(', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'<!--.*?-->', '', sanitized, flags=re.DOTALL)

    # Additional sanitization for SQL injection prevention
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
        r"(--.*)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(';|';|--|\b(OR|AND)\b)"
    ]

    for pattern in sql_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

    return sanitized.strip()


def validate_user_id(user_id: Optional[str]) -> bool:
    """
    Validate user ID format.
    """
    if user_id is None:
        return True
    # Basic validation: alphanumeric, underscore, hyphen, 1-50 chars
    pattern = r'^[a-zA-Z0-9_-]{1,50}$'
    return bool(re.match(pattern, user_id))


def validate_session_id(session_id: Optional[str]) -> bool:
    """
    Validate session ID format.
    """
    if session_id is None:
        return True
    # Basic validation: alphanumeric, underscore, hyphen, 1-100 chars
    pattern = r'^[a-zA-Z0-9_-]{1,100}$'
    return bool(re.match(pattern, session_id))