"""
FastAPI Gateway Main Application

Domain-routing API gateway.
"""

import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .router import create_router
from ..common.logging_utils import get_logger
from ..common.schemas import HealthResponse

logger = get_logger(__name__)

# Global state
start_time = time.time()
models_loaded: List[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting LLM Portfolio Platform API")
    logger.info("Loading models...")

    # TODO: Load models here
    # Example: vllm_engine.load()
    models_loaded.append("base_model")

    logger.info("Models loaded successfully")

    yield

    # Shutdown
    logger.info("Shutting down...")
    # Example: vllm_engine.unload()


app = FastAPI(
    title="LLM Portfolio Platform API",
    description="E-commerce Customer Service + Financial RAG System",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(create_router())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
