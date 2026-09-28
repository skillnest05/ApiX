"""
Main entrypoint for APIx FastAPI Application.
Exposes 'app' for uvicorn apix.api.main:app.
"""

from apix.api.app import app

__all__ = ["app"]
