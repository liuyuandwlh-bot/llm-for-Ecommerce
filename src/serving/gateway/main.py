"""
FastAPI Gateway Main Application

Domain-routing API gateway.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.common.logging_utils import get_logger
from src.common.schemas import HealthResponse

logger = get_logger(__name__)

start_time = time.time()


def create_app() -> FastAPI:
    """Create FastAPI application."""

    models_loaded: list[str] = []
    backend_mode: Literal["mock", "real"] = (
        "real" if os.environ.get("LLM_PORTFOLIO_REAL_BACKEND") == "1" else "mock"
    )
    backend_ready = False
    cache_hits = 0
    cache_misses = 0

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan events."""
        nonlocal models_loaded, backend_ready
        logger.info("Starting LLM Portfolio Platform API (mode=%s)", backend_mode)

        if backend_mode == "real":
            # Real backend wiring belongs in src.serving.gateway.bootstrap.
            # Loading a real model is out of scope for this round.
            backend_ready = False
            logger.info("Real backend requested but no model is loaded; using mock.")
        yield
        logger.info("Shutting down...")

    app = FastAPI(
        title="LLM Portfolio Platform API",
        description="E-commerce Customer Service + Financial RAG System",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS - configurable via environment (defaults to a safe local-only set).
    allowed_origins_env = os.environ.get("LLM_PORTFOLIO_ALLOWED_ORIGINS")
    if allowed_origins_env:
        allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    else:
        allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Include routers
    from src.serving.gateway.router import create_router
    app.include_router(create_router())

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        status: Literal["healthy", "degraded", "unhealthy"] = (
            "degraded" if backend_mode == "real" and not backend_ready else "healthy"
        )
        return HealthResponse(
            status=status,
            version="0.2.0",
            uptime_seconds=time.time() - start_time,
            mode=backend_mode,
            models_loaded=models_loaded,
            backend_ready=backend_ready,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP error: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPError",
                "message": exc.detail,
                "status_code": exc.status_code,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
            },
        )

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
