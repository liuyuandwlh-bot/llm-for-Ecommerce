"""
Common Utilities Module

Shared utilities for logging, versioning, PII handling, and schemas.
"""

from .logging_utils import setup_logging, get_logger
from .versioning import (
    compute_data_hash,
    compute_model_hash,
    VersionInfo,
)
from .pii import PIIDetector, mask_pii
from .schemas import (
    CustomerServiceRequest,
    CustomerServiceResponse,
    FinanceRAGRequest,
    FinanceRAGResponse,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "compute_data_hash",
    "compute_model_hash",
    "VersionInfo",
    "PIIDetector",
    "mask_pii",
    "CustomerServiceRequest",
    "CustomerServiceResponse",
    "FinanceRAGRequest",
    "FinanceRAGResponse",
]
