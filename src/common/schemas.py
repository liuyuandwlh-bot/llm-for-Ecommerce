"""
API Schemas

Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Any


# Customer Service Schemas
class CustomerServiceRequest(BaseModel):
    """Customer service request schema."""
    query: str = Field(..., description="User query")
    domain: Literal["ecommerce", "finance"] = Field(default="ecommerce")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Conversation history")
    user_id: Optional[str] = Field(default=None, description="User identifier (for tracking)")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class CustomerServiceResponse(BaseModel):
    """Customer service response schema."""
    response: str = Field(..., description="Model response")
    intent: Optional[str] = Field(default=None, description="Detected intent")
    slots: Optional[Dict[str, Any]] = Field(default=None, description="Extracted slots")
    requires_human: bool = Field(default=False, description="Whether to escalate")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    latency_ms: float = Field(..., description="Response latency in milliseconds")


# Finance RAG Schemas
class FinanceRAGRequest(BaseModel):
    """Finance RAG request schema."""
    query: str = Field(..., description="User query")
    domain: Literal["finance"] = Field(default="finance")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters (company, year, etc.)"
    )
    include_calculations: bool = Field(default=True, description="Include numerical calculations")
    max_citations: int = Field(default=5, ge=1, le=10, description="Max citations to return")


class Citation(BaseModel):
    """A citation in the response."""
    doc_id: str
    page: int
    evidence: str
    score: Optional[float] = None


class FinanceRAGResponse(BaseModel):
    """Finance RAG response schema."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
    limitations: List[str] = Field(default_factory=list, description="Known limitations")
    calculations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Calculations performed")
    latency_ms: float = Field(..., description="Response latency in milliseconds")


# Health Check Schemas
class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: float
    models_loaded: List[str] = Field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0


# Error Schemas
class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None)
    request_id: Optional[str] = Field(default=None)


# Batch Request Schemas
class BatchCustomerServiceRequest(BaseModel):
    """Batch customer service request."""
    requests: List[CustomerServiceRequest] = Field(..., max_length=100)


class BatchCustomerServiceResponse(BaseModel):
    """Batch customer service response."""
    responses: List[CustomerServiceResponse]
    total_latency_ms: float
