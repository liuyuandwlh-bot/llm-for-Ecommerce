"""
API Router

Domain routing and endpoint definitions.
"""

import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from src.common.logging_utils import get_logger
from src.common.pii import mask_pii_in_text
from src.common.schemas import (
    CustomerServiceRequest,
    CustomerServiceResponse,
    FinanceRAGRequest,
    FinanceRAGResponse,
)

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

        Returns mock response when model not loaded.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]

        # Log request (with masked PII)
        masked_query = mask_pii_in_text(request.query)
        logger.info(f"[{request_id}] Customer service request: %s", masked_query)

        # Validate domain
        if request.domain not in ["ecommerce", "finance"]:
            raise HTTPException(
                status_code=400, detail="Only 'ecommerce' or 'finance' domain supported"
            )

        try:
            # TODO: Call actual model
            # For now, return a mock response
            # In production, this would route to:
            # - E-commerce: SFT model with LoRA adapter
            # - Finance: RAG pipeline

            response_text = f"[电商客服-MOCK] 已收到您的问题：{masked_query}"

            return CustomerServiceResponse(
                response=response_text,
                intent="mock_intent",
                slots={},
                requires_human=False,
                confidence=0.8,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"[{request_id}] Customer service error: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/api/v1/finance-rag", response_model=FinanceRAGResponse)
    async def finance_rag(request: FinanceRAGRequest, req: Request):
        """
        Financial RAG endpoint.

        Returns answers with citations from official disclosures.
        Mock response when RAG pipeline not available.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]

        # Log request (with masked PII)
        masked_query = mask_pii_in_text(request.query)
        logger.info(f"[{request_id}] Finance RAG request: %s", masked_query)

        # Validate domain
        if request.domain != "finance":
            raise HTTPException(status_code=400, detail="Only 'finance' domain supported")

        try:
            # TODO: Call actual RAG pipeline
            # In production: document retrieval -> reranking -> answer generation

            response_text = "[金融RAG-MOCK] 已检索相关信息回答您的问题。"

            return FinanceRAGResponse(
                answer=response_text,
                citations=[],
                confidence="medium",
                limitations=["This is a mock response - RAG pipeline not loaded"],
                calculations=[],
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"[{request_id}] Finance RAG error: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post("/api/v1/batch")
    async def batch_predict(requests: list[CustomerServiceRequest], req: Request):
        """
        Batch prediction endpoint.

        Limited to 100 requests per batch.
        """
        # Limit batch size
        if len(requests) > 100:
            raise HTTPException(status_code=400, detail="Batch size limited to 100 requests")

        responses = []
        for request in requests:
            result = await customer_service(request, req)
            responses.append(result)

        return {"responses": responses, "count": len(responses)}

    return router
