"""
Centralized Configuration and Security Settings for APIx Platform.
Manages environment variables, security credentials, database URLs,
CORS policies, and production deployment parameters.
"""

import os
from typing import List, Optional


class Settings:
    """Production application settings with environment variable overrides."""

    # Environment & Debugging
    ENV: str = os.getenv("APIX_ENV", "production").lower()
    DEBUG: bool = os.getenv("APIX_DEBUG", "false").lower() in ("true", "1", "yes")

    # Server Binding
    HOST: str = os.getenv("APIX_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT") or os.getenv("APIX_PORT", "8000"))

    # Database Connection
    DATABASE_URL: str = (
        os.getenv("APIX_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or "sqlite:///apix.db"
    )

    # API Security & Authentication
    # If API_KEY is set in environment, protected endpoints require X-API-Key or Bearer token
    API_KEY: Optional[str] = os.getenv("APIX_API_KEY", "").strip() or None
    SECRET_KEY: str = os.getenv(
        "APIX_SECRET_KEY", "apix-secure-session-key-change-in-production-env"
    )

    # Cross-Origin Resource Sharing (CORS)
    _default_origins = "http://localhost:3000,http://127.0.0.1:8000,http://localhost:8000"
    ALLOWED_ORIGINS: List[str] = [
        orig.strip()
        for orig in os.getenv("APIX_ALLOWED_ORIGINS", _default_origins).split(",")
        if orig.strip()
    ]

    # Rate Limiting (Requests per minute per client IP)
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("APIX_RATE_LIMIT_PER_MINUTE", "120"))

    # Security Headers & TLS
    ENABLE_SECURITY_HEADERS: bool = os.getenv(
        "APIX_SECURE_HEADERS", "true"
    ).lower() in ("true", "1", "yes")

    # Interactive Documentation Toggle (/docs, /redoc)
    ENABLE_DOCS: bool = os.getenv("APIX_ENABLE_DOCS", "true").lower() in (
        "true", "1", "yes"
    )


settings = Settings()
