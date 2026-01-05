import asyncio
import time
from collections import defaultdict, deque
from typing import Dict
from ..config.settings import settings


class RateLimiter:
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, identifier: str = "default") -> bool:
        """
        Check if a request is allowed based on rate limits.

        Args:
            identifier: Identifier for the client/user making the request

        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()
        # Remove requests that are outside the time window
        while (self.requests[identifier] and
               current_time - self.requests[identifier][0] > self.time_window):
            self.requests[identifier].popleft()

        # Check if we've exceeded the limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Add current request
        self.requests[identifier].append(current_time)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=50, time_window=60)  # Adjust as needed based on API limits


class RequestQueue:
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = set()

    async def add_request(self, request_id: str, coro):
        """Add a request to the queue."""
        if self.queue.full():
            raise Exception("Request queue is full")

        await self.queue.put((request_id, coro))

    async def get_next_request(self):
        """Get the next request from the queue."""
        return await self.queue.get()


# Global request queue instance
request_queue = RequestQueue()