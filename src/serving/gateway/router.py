"""
API Router

Domain routing and endpoint definitions.
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from typing import Optional
import time

from ..common.logging_utils import get_logger
from ..common.schemas import (
    CustomerServiceRequest,
    CustomerServiceResponse,
    FinanceRAGRequest,
    FinanceRAGResponse,
    ErrorResponse,
)
from ..common.pii import mask_pii_in_text

logger = get_logger(__name__)


def create_router() -> APIRouter:
    """Create the main API router."""
    router = APIRouter()

    @router.post("/api/v1/customer-service", response_model=CustomerServiceResponse)
    async def customer_service(request: CustomerServiceRequest, req: Request):
        """
        Customer service endpoint.

        Routes to:
        - E-commerce customer service (with LoRA adapter)
        - Financial Q&A (with RAG)
        """
        start_time = time.time()

        # Validate domain
        if request.domain != "ecommerce":
            raise HTTPException(status_code=400, detail="Only 'ecommerce' domain supported in this version")

        try:
            # TODO: Call actual model
            # For now, return a placeholder response

            response_text = f"[电商客服] 已收到您的问题：{request.query}"

            return CustomerServiceResponse(
                response=response_text,
                intent="general_query",
                slots={},
                requires_human=False,
                confidence=0.8,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Customer service error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/v1/finance-rag", response_model=FinanceRAGResponse)
    async def finance_rag(request: FinanceRAGRequest, req: Request):
        """
        Financial RAG endpoint.

        Returns answers with citations from official disclosures.
        """
        start_time = time.time()

        # Validate domain
        if request.domain != "finance":
            raise HTTPException(status_code=400, detail="Only 'finance' domain supported")

        try:
            # TODO: Call actual RAG pipeline

            response_text = f"[金融RAG] 已检索相关信息回答您的问题。"

            return FinanceRAGResponse(
                answer=response_text,
                citations=[],
                confidence="medium",
                limitations=[],
                calculations=[],
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Finance RAG error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/v1/batch")
    async def batch_predict(requests: list[CustomerServiceRequest], req: Request):
        """Batch prediction endpoint."""
        responses = []

        for request in requests:
            result = await customer_service(request, req)
            responses.append(result)

        return {"responses": responses}

    return router
