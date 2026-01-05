import time
import asyncio
from typing import Dict, Any, Callable, Awaitable
from collections import deque
import logging
from datetime import datetime, timedelta


class PerformanceMonitor:
    def __init__(self, window_size: int = 1000):
        self.request_times = deque(maxlen=window_size)
        self.error_count = 0
        self.request_count = 0
        self.start_time = time.time()

    def record_request(self, start_time: float, end_time: float, success: bool = True):
        """Record a request with its duration."""
        duration = end_time - start_time
        self.request_times.append(duration)
        self.request_count += 1

        if not success:
            self.error_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.request_times:
            return {
                "requests_per_second": 0,
                "avg_response_time": 0,
                "p95_response_time": 0,
                "p99_response_time": 0,
                "error_rate": 0,
                "uptime": time.time() - self.start_time
            }

        # Calculate requests per second
        duration = time.time() - self.start_time
        requests_per_second = self.request_count / duration if duration > 0 else 0

        # Calculate response time metrics
        sorted_times = sorted(list(self.request_times))
        avg_response_time = sum(self.request_times) / len(self.request_times)

        # Calculate percentiles
        n = len(sorted_times)
        p95_idx = int(0.95 * n) - 1 if n > 0 else 0
        p99_idx = int(0.99 * n) - 1 if n > 0 else 0

        p95_response_time = sorted_times[min(p95_idx, n-1)] if n > 0 else 0
        p99_response_time = sorted_times[min(p99_idx, n-1)] if n > 0 else 0

        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0

        return {
            "requests_per_second": requests_per_second,
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "error_rate": error_rate,
            "uptime": time.time() - self.start_time,
            "total_requests": self.request_count,
            "total_errors": self.error_count
        }

    def is_performance_degraded(self) -> bool:
        """Check if performance is degraded based on thresholds."""
        metrics = self.get_metrics()

        # Check if average response time is greater than 2 seconds
        if metrics["avg_response_time"] > 2.0:
            return True

        # Check if error rate is too high (e.g., >5%)
        if metrics["error_rate"] > 0.05:
            return True

        return False


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_endpoint(func: Callable) -> Callable:
    """Decorator to monitor endpoint performance."""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            performance_monitor.record_request(start_time, end_time, success)
    return wrapper