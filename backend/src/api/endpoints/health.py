from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime
import asyncio

from ...services.qdrant_service import qdrant_service
from ...services.postgres_service import postgres_service
from ...services.cohere_service import cohere_service

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify the status of all services.
    """
    # Check each service's health
    qdrant_healthy = True
    postgres_healthy = True
    cohere_healthy = True

    try:
        # Test Qdrant connection
        collections = await qdrant_service.client.get_collections()
    except Exception:
        qdrant_healthy = False

    try:
        # Test Postgres connection
        # Just try to get a connection from the pool
        pool = await postgres_service.db_manager.get_connection()
    except Exception:
        postgres_healthy = False

    try:
        # Test Cohere connection with a simple request
        from ...config.settings import settings
        import cohere
        client = cohere.Client(settings.cohere_api_key)
        response = client.generate(
            model='command',
            prompt='Hello',
            max_tokens=5
        )
    except Exception:
        cohere_healthy = False

    # Overall health status
    overall_healthy = qdrant_healthy and postgres_healthy and cohere_healthy

    health_status = {
        "status": "healthy" if overall_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": {
            "qdrant": qdrant_healthy,
            "postgres": postgres_healthy,
            "cohere": cohere_healthy
        }
    }

    if not overall_healthy:
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check endpoint to verify the service is ready to accept requests.
    """
    return {"status": "ready"}