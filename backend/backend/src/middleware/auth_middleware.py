from fastapi import HTTPException, status, Request
from ..config.settings import settings
import secrets


def api_key_auth(request: Request):
    """
    Simple API key authentication middleware
    """
    api_key = request.headers.get("X-API-Key")
    
    if not api_key or not secrets.compare_digest(api_key, settings.secret_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )