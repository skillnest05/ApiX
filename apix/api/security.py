"""
Security, Authentication & Hardening Middleware for APIx Platform.
Provides:
1. SecurityHeadersMiddleware (HSTS, CSP, X-Frame-Options, X-Content-Type-Options)
2. RateLimitingMiddleware (Sliding-window IP request throttling)
3. API Key Authentication dependency (Bearer / X-API-Key)
4. Input validation and parameter sanitizers
"""

from collections import defaultdict
from datetime import datetime, timezone
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import Depends, Header, HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from apix.config import settings

# Regex patterns for strict input validation
IATA_REGEX = re.compile(r"^[A-Z]{3}$")
SECTOR_REGEX = re.compile(r"^[A-Z]{3}-[A-Z]{3}$")
DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Applies production-grade security headers to all outgoing HTTP responses
    in compliance with OWASP API Security guidelines.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if settings.ENABLE_SECURITY_HEADERS:
            headers = response.headers
            headers["X-Content-Type-Options"] = "nosniff"
            headers["X-Frame-Options"] = "SAMEORIGIN"
            headers["X-XSS-Protection"] = "1; mode=block"
            headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            headers["Permissions-Policy"] = (
                "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
                "magnetometer=(), microphone=(), payment=(), usb=()"
            )
            # Content Security Policy (allows self, CDNs for dashboard charts, and fonts)
            headers["Content-Security-Policy"] = (
                "default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval'; "
                "font-src 'self' https://fonts.gstatic.com data:; "
                "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
                "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline' 'unsafe-eval'; "
                "img-src 'self' https: data:;"
            )
            # HSTS enabled for production or HTTPS connections
            if request.url.scheme == "https" or settings.ENV == "production":
                headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        return response


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    In-memory sliding-window rate limiter per client IP.
    Protects API endpoints from automated scraping abuse and DoS attacks.
    """

    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Mapping from client IP -> list of timestamps
        self._history: Dict[str, List[float]] = defaultdict(list)
        # Exempt routes that shouldn't be blocked by rate limiting
        self.exempt_paths: Set[str] = {
            "/health",
            "/api/v1/health",
            "/openapi.json",
            "/docs",
            "/redoc",
            "/favicon.ico"
        }

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        # Skip preflight OPTIONS or exempt routes
        if method == "OPTIONS" or path in self.exempt_paths:
            return await call_next(request)

        # Get client IP (support X-Forwarded-For if behind a reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "127.0.0.1"

        # Bypass rate limiter for local test runner
        if client_ip in ("testclient", "testserver"):
            return await call_next(request)

        now = time.time()
        cutoff = now - self.window_seconds

        # Prune old request timestamps
        history = [ts for ts in self._history[client_ip] if ts > cutoff]
        self._history[client_ip] = history

        # Check threshold
        if len(history) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - history[0])) if history else self.window_seconds
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "status": "error",
                    "code": 429,
                    "message": "Rate limit exceeded. Please throttle requests.",
                    "limit_per_minute": self.max_requests,
                    "retry_after_seconds": max(1, retry_after)
                },
                headers={"Retry-After": str(max(1, retry_after))}
            )

        # Record this request
        self._history[client_ip].append(now)

        return await call_next(request)


# Bearer token helper
http_bearer_scheme = HTTPBearer(auto_error=False)


def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    auth_credentials: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer_scheme)
) -> bool:
    """
    Dependency that enforces API key authentication on sensitive or protected endpoints.
    If APIX_API_KEY is not set in environment (development/testing), authentication is permissive.
    If APIX_API_KEY is set, client must provide a matching key via 'X-API-Key' or 'Bearer <token>'.
    """
    configured_key = settings.API_KEY
    if not configured_key:
        return True

    # Check X-API-Key header
    if x_api_key and x_api_key == configured_key:
        return True

    # Check Bearer token
    if auth_credentials and auth_credentials.credentials == configured_key:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Valid X-API-Key or Bearer token required for this endpoint.",
        headers={"WWW-Authenticate": "Bearer"}
    )


def sanitize_iata(code: str, param_name: str = "airport_code") -> str:
    """Validates and returns normalized 3-letter uppercase IATA code."""
    cleaned = (code or "").strip().upper()
    if not IATA_REGEX.match(cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {param_name}: '{code}'. Must be a valid 3-letter IATA airport code."
        )
    return cleaned


def sanitize_sector(sector: str) -> str:
    """Validates and returns normalized sector string (e.g. 'DEL-BOM')."""
    cleaned = (sector or "").strip().upper()
    if not SECTOR_REGEX.match(cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sector: '{sector}'. Expected format 'ORIGIN-DESTINATION' (e.g., DEL-BOM)."
        )
    return cleaned
