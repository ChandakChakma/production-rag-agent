"""
FastAPI middleware stack:
  1. RequestIDMiddleware — injects X-Request-ID into every request/response
  2. RateLimitMiddleware — sliding window rate limiter (in-memory, per IP)
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.unbind_contextvars("request_id")
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter. Exempt: /health, /metrics."""

    _EXEMPT = {"/health", "/metrics"}

    def __init__(self, app, requests_per_minute: int = 100) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._window: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._EXEMPT:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._window[client_ip]

        # Remove timestamps older than 60 seconds
        while window and window[0] < now - 60:
            window.popleft()

        if len(window) >= self._rpm:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again in a minute."},
                headers={"Retry-After": "60"},
            )

        window.append(now)
        return await call_next(request)
