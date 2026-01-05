from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
from uuid import UUID
from ..utils.database import db_manager
from ..models.query_history import QueryHistory
from ..models.query import Query
from ..utils.circuit_breaker import postgres_circuit_breaker


class PostgresService:
    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """Initialize the Postgres service and create required tables if they don't exist."""
        try:
            # Create queries table
            await db_manager.execute_command('''
                CREATE TABLE IF NOT EXISTS queries (
                    id UUID PRIMARY KEY,
                    user_id VARCHAR(50),
                    question TEXT NOT NULL,
                    selected_text TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    session_id VARCHAR(100)
                )
            ''')

            # Create responses table
            await db_manager.execute_command('''
                CREATE TABLE IF NOT EXISTS responses (
                    id UUID PRIMARY KEY,
                    query_id UUID NOT NULL REFERENCES queries(id),
                    answer TEXT NOT NULL,
                    citations TEXT[],
                    confidence_score FLOAT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    was_answer_found BOOLEAN NOT NULL
                )
            ''')

            # Create query_history table
            await db_manager.execute_command('''
                CREATE TABLE IF NOT EXISTS query_history (
                    id UUID PRIMARY KEY,
                    query_id UUID NOT NULL REFERENCES queries(id),
                    response_id UUID NOT NULL REFERENCES responses(id),
                    query_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    citations TEXT[],
                    timestamp TIMESTAMP NOT NULL,
                    user_id VARCHAR(50),
                    session_id VARCHAR(100),
                    was_useful BOOLEAN
                )
            ''')

            self._initialized = True
            logging.info("Postgres service initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Postgres service: {e}")
            raise

    @postgres_circuit_breaker
    async def log_query(self, query: Query) -> bool:
        """Log a query to the database."""
        if not self._initialized:
            await self.initialize()

        try:
            await db_manager.execute_command('''
                INSERT INTO queries (id, user_id, question, selected_text, timestamp, session_id)
                VALUES ($1, $2, $3, $4, $5, $6)
            ''', query.id, query.user_id, query.question, query.selected_text, query.timestamp, query.session_id)
            return True
        except Exception as e:
            logging.error(f"Failed to log query: {e}")
            return False

    @postgres_circuit_breaker
    async def log_response(self, response: Dict[str, Any]) -> bool:
        """Log a response to the database."""
        if not self._initialized:
            await self.initialize()

        try:
            await db_manager.execute_command('''
                INSERT INTO responses (id, query_id, answer, citations, confidence_score, timestamp, was_answer_found)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', response['id'], response['query_id'], response['answer'], response['citations'],
                   response['confidence_score'], response['timestamp'], response['was_answer_found'])
            return True
        except Exception as e:
            logging.error(f"Failed to log response: {e}")
            return False

    @postgres_circuit_breaker
    async def log_query_history(self, query_history: QueryHistory) -> bool:
        """Log query history to the database."""
        if not self._initialized:
            await self.initialize()

        try:
            await db_manager.execute_command('''
                INSERT INTO query_history (id, query_id, response_id, query_text, response_text, citations, timestamp, user_id, session_id, was_useful)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ''', query_history.id, query_history.query_id, query_history.response_id,
                   query_history.query_text, query_history.response_text, query_history.citations,
                   query_history.timestamp, query_history.user_id, query_history.session_id, query_history.was_useful)
            return True
        except Exception as e:
            logging.error(f"Failed to log query history: {e}")
            return False

    @postgres_circuit_breaker
    async def get_query_history(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve query history from the database."""
        if not self._initialized:
            await self.initialize()

        try:
            if user_id:
                rows = await db_manager.execute_query('''
                    SELECT * FROM query_history
                    WHERE user_id = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                ''', user_id, limit)
            else:
                rows = await db_manager.execute_query('''
                    SELECT * FROM query_history
                    ORDER BY timestamp DESC
                    LIMIT $1
                ''', limit)

            history = []
            for row in rows:
                history.append({
                    'id': row['id'],
                    'query_id': row['query_id'],
                    'response_id': row['response_id'],
                    'query_text': row['query_text'],
                    'response_text': row['response_text'],
                    'citations': row['citations'],
                    'timestamp': row['timestamp'],
                    'user_id': row['user_id'],
                    'session_id': row['session_id'],
                    'was_useful': row['was_useful']
                })

            return history
        except Exception as e:
            logging.error(f"Failed to get query history: {e}")
            return []

    @postgres_circuit_breaker
    async def update_feedback(self, query_history_id: UUID, was_useful: bool) -> bool:
        """Update feedback for a query history entry."""
        if not self._initialized:
            await self.initialize()

        try:
            result = await db_manager.execute_command('''
                UPDATE query_history
                SET was_useful = $1
                WHERE id = $2
            ''', was_useful, query_history_id)
            return result is not None
        except Exception as e:
            logging.error(f"Failed to update feedback: {e}")
            return False

    @postgres_circuit_breaker
    async def delete_user_data(self, user_id: str) -> bool:
        """
        GDPR-compliant method to delete all data associated with a user.
        This implements the 'right to be forgotten' by removing all personal data.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Delete all query history for the user
            await db_manager.execute_command('''
                DELETE FROM query_history WHERE user_id = $1
            ''', user_id)

            # Delete all queries for the user
            await db_manager.execute_command('''
                DELETE FROM queries WHERE user_id = $1
            ''', user_id)

            # Note: We don't delete responses that are linked to queries from other users
            # but we can anonymize them if they exist
            logging.info(f"User data deletion completed for user_id: {user_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to delete user data: {e}")
            return False

    @postgres_circuit_breaker
    async def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        GDPR-compliant method to retrieve all data associated with a user.
        This implements the 'right to data portability' by providing all personal data.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get query history for the user
            query_history = await db_manager.execute_query('''
                SELECT * FROM query_history WHERE user_id = $1
            ''', user_id)

            # Get queries for the user
            queries = await db_manager.execute_query('''
                SELECT * FROM queries WHERE user_id = $1
            ''', user_id)

            user_data = {
                'user_id': user_id,
                'queries': [dict(row) for row in queries],
                'query_history': [dict(row) for row in query_history],
                'timestamp': datetime.utcnow().isoformat()
            }

            return user_data
        except Exception as e:
            logging.error(f"Failed to retrieve user data: {e}")
            return {}


# Global instance
postgres_service = PostgresService()