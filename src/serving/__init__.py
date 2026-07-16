"""
FastAPI Gateway

Domain-routing API gateway for customer service and financial RAG.
"""

from .main import app
from .router import create_router

__all__ = ["app", "create_router"]
