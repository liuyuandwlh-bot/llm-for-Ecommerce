"""
API Schemas

Pydantic models for API request/response validation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# Customer Service Schemas
class CustomerServiceRequest(BaseModel):
    """Customer service request schema."""

    query: str = Field(..., description="User query")
    domain: Literal["ecommerce", "finance"] = Field(default="ecommerce")
    history: list[dict[str, str]] | None = Field(default=None, description="Conversation history")
    user_id: str | None = Field(default=None, description="User identifier (for tracking)")
    session_id: str | None = Field(default=None, description="Session identifier")


class CustomerServiceResponse(BaseModel):
    """Customer service response schema."""

    response: str = Field(..., description="Model response")
    intent: str | None = Field(default=None, description="Detected intent")
    slots: dict[str, Any] | None = Field(default=None, description="Extracted slots")
    requires_human: bool = Field(default=False, description="Whether to escalate")
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float = Field(..., description="Response latency in milliseconds")


# Finance RAG Schemas
class FinanceRAGRequest(BaseModel):
    """Finance RAG request schema."""

    query: str = Field(..., description="User query")
    domain: Literal["finance"] = Field(default="finance")
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata filters (company, year, etc.)"
    )
    include_calculations: bool = Field(default=True, description="Include numerical calculations")
    max_citations: int = Field(default=5, ge=1, le=10, description="Max citations to return")


class Citation(BaseModel):
    """A citation in the response."""

    doc_id: str
    page: int
    evidence: str
    score: float | None = None


class FinanceRAGResponse(BaseModel):
    """Finance RAG response schema."""

    answer: str = Field(..., description="Generated answer")
    citations: list[Citation] = Field(default_factory=list, description="Source citations")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
    limitations: list[str] = Field(default_factory=list, description="Known limitations")
    calculations: list[dict[str, Any]] | None = Field(
        default=None, description="Calculations performed"
    )
    latency_ms: float = Field(..., description="Response latency in milliseconds")


# Health Check Schemas
class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    mode: Literal["mock", "real"] = "mock"
    models_loaded: list[str] = Field(default_factory=list)
    backend_ready: bool = False
    cache_hits: int = 0
    cache_misses: int = 0


# Error Schemas
class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(default=None)
    request_id: str | None = Field(default=None)


# Batch Request Schemas
class BatchCustomerServiceRequest(BaseModel):
    """Batch customer service request."""

    requests: list[CustomerServiceRequest] = Field(..., max_length=100)


class BatchCustomerServiceResponse(BaseModel):
    """Batch customer service response."""

    responses: list[CustomerServiceResponse]
    total_latency_ms: float
