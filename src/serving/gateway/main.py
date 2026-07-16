"""
FastAPI Gateway Main Application

Domain-routing API gateway.
"""

import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.common.logging_utils import get_logger
from src.common.schemas import HealthResponse

logger = get_logger(__name__)

# Global state
start_time = time.time()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    
    models_loaded: List[str] = []
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan events."""
        nonlocal models_loaded
        
        # Startup
        logger.info("Starting LLM Portfolio Platform API")
        logger.info("Loading models...")
        
        # Model loading is deferred to actual endpoints
        # to support both mock and real backends
        models_loaded.append("placeholder")
        
        logger.info("Models loaded: %s", models_loaded)
        
        yield
        
        # Shutdown
        logger.info("Shutting down...")

    app = FastAPI(
        title="LLM Portfolio Platform API",
        description="E-commerce Customer Service + Financial RAG System",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS - configure properly for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure for production
        allow_credentials=True,  # TODO: Set specific origins in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from src.serving.gateway.router import create_router
    app.include_router(create_router())

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.2.0",
            uptime_seconds=time.time() - start_time,
            models_loaded=models_loaded,
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
