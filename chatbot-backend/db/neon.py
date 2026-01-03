import asyncpg
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

NEON_DB_URL = os.getenv("DATABASE_URL")

async def get_db_connection():
    """
    Get a connection to the Neon database
    """
    try:
        conn = await asyncpg.connect(NEON_DB_URL)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Neon database: {str(e)}")
        raise e