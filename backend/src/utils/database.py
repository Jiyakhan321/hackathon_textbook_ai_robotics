import asyncpg
from typing import Optional
from ..config.settings import settings
import logging


class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize the database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                dsn=settings.neon_url,
                min_size=10,  # Increase min size for better concurrency
                max_size=50,  # Increase max size for better concurrency
                command_timeout=60,
                max_queries=50000,  # Maximum number of queries per connection
                max_inactive_connection_lifetime=300.0  # 5 minutes
            )
            logging.info("Database connection pool created successfully")
        except Exception as e:
            logging.error(f"Failed to create database connection pool: {e}")
            raise

    async def disconnect(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            logging.info("Database connection pool closed")

    async def get_connection(self):
        """Get a connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.pool

    async def execute_query(self, query: str, *args):
        """Execute a query with the given parameters."""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def execute_query_row(self, query: str, *args):
        """Execute a query and return a single row."""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute_command(self, command: str, *args):
        """Execute a command (INSERT, UPDATE, DELETE) and return affected rows."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(command, *args)


# Global instance
db_manager = DatabaseManager()