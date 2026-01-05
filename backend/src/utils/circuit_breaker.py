import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Tripped, requests blocked
    HALF_OPEN = "half_open"  # Testing if failure condition is resolved


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

        # Lock to prevent race conditions
        self._lock = asyncio.Lock()

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self._lock:
                if self.state == CircuitState.OPEN:
                    # Check if it's time to retry
                    if time.time() - self.last_failure_time >= self.recovery_timeout:
                        self.state = CircuitState.HALF_OPEN
                    else:
                        raise Exception(f"Circuit breaker is OPEN. Call to {func.__name__} blocked.")

            try:
                result = await func(*args, **kwargs)

                async with self._lock:
                    if self.state == CircuitState.HALF_OPEN or self.state == CircuitState.OPEN:
                        # Success after failure, reset the circuit
                        self.failure_count = 0
                        self.state = CircuitState.CLOSED
                        self.last_failure_time = None

                return result

            except self.expected_exception as e:
                async with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    if self.failure_count >= self.failure_threshold:
                        self.state = CircuitState.OPEN

                raise e

        return wrapper


# Global circuit breaker instances for different services
cohere_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception
)

qdrant_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception
)

postgres_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception
)