"""Lightweight in-memory API metrics collection and middleware.

No external dependencies — uses only stdlib and Starlette internals.
Metrics reset on process restart (by design for a simple monitoring tool).
"""

import collections
import json
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from starlette.requests import Request
from starlette.responses import Response

# Paths to exclude from metrics recording (avoid self-referential noise)
_EXCLUDED_PATHS = {"/metrics", "/dashboard", "/docs", "/openapi.json", "/redoc", "/test-data"}


class MetricsStore:
    """Thread-safe in-memory metrics store."""

    def __init__(self, history_size: int = 1000, price_history_size: int = 500):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency_ms = 0.0
        self.status_codes = {}
        self.endpoint_counts = {}
        self.request_log = collections.deque(maxlen=history_size)
        self.price_history = collections.deque(maxlen=price_history_size)

    def record(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        predicted_price: Optional[float] = None,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if status_code >= 400:
                self.total_errors += 1

            key = str(status_code)
            self.status_codes[key] = self.status_codes.get(key, 0) + 1
            self.endpoint_counts[path] = self.endpoint_counts.get(path, 0) + 1

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
            }
            if predicted_price is not None:
                entry["predicted_price"] = predicted_price
                self.price_history.append(predicted_price)

            self.request_log.append(entry)

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = (
                round(self.total_latency_ms / self.total_requests, 2)
                if self.total_requests > 0
                else 0.0
            )
            error_rate = (
                round(self.total_errors / self.total_requests * 100, 2)
                if self.total_requests > 0
                else 0.0
            )
            return {
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "avg_latency_ms": avg_latency,
                "error_rate_pct": error_rate,
                "status_codes": dict(self.status_codes),
                "endpoint_counts": dict(self.endpoint_counts),
                "price_history": list(self.price_history),
                "recent_requests": list(self.request_log)[-50:],
            }


# Module-level singleton
store = MetricsStore()


async def metrics_middleware(request: Request, call_next) -> Response:
    """Capture request metrics for every API call (except excluded paths)."""
    if request.url.path in _EXCLUDED_PATHS:
        return await call_next(request)

    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000

    predicted_price = None

    # Extract predicted_price from /predict responses
    if request.url.path == "/predict" and request.method == "POST" and response.status_code == 200:
        body_bytes = b""
        async for chunk in response.body_iterator:
            body_bytes += chunk
        try:
            body_json = json.loads(body_bytes)
            predicted_price = body_json.get("predicted_price")
        except (json.JSONDecodeError, AttributeError):
            pass
        # Re-wrap response so the client still receives the body
        response = Response(
            content=body_bytes,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    store.record(
        method=request.method,
        path=str(request.url.path),
        status_code=response.status_code,
        latency_ms=latency_ms,
        predicted_price=predicted_price,
    )

    return response
