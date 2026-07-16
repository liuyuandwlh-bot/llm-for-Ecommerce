"""
Financial RAG Module

Handles:
- Document ingestion (official disclosures, annual reports)
- PDF parsing with layout preservation
- Chunking with table protection
- Retrieval with hybrid BM25 + Dense + RRF
- Reranking with cross-encoder
- Answer generation with citations
"""

from .document_ingest import DocumentIngestor, DocumentManifest
from .pdf_parser import PDFParser, parse_batch
from .chunker import ChunkBuilder, ChunkConfig
from .retriever import HybridRetriever, RetrievalResult
from .reranker import CrossEncoderReranker
from .answer_engine import AnswerEngine, Answer

__all__ = [
    "DocumentIngestor",
    "DocumentManifest",
    "PDFParser",
    "parse_batch",
    "ChunkBuilder",
    "ChunkConfig",
    "HybridRetriever",
    "RetrievalResult",
    "CrossEncoderReranker",
    "AnswerEngine",
    "Answer",
]
