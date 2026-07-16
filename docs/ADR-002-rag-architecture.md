# ADR-002: RAG Architecture Decision

## Status
Accepted

## Context
We need to design the retrieval architecture for the financial RAG system. Key considerations:
- 200-300 PDF documents (annual reports, policy documents)
- Need page-level citation traceability
- Multi-table financial documents
- Mix of sparse (BM25) and dense retrieval

## Decision
Use a hybrid retrieval architecture with:
1. BM25 + Dense (Qwen3-Embedding-0.6B or BGE-M3)
2. Reciprocal Rank Fusion (RRF) for combination
3. Cross-encoder Reranker (Qwen3-Reranker-0.6B or BGE-reranker-v2-m3)
4. Parent-child chunking with table protection

### Why Not Single Dense?

| Approach | Pros | Cons |
|----------|------|------|
| Dense-only | Simple, semantic match | Misses exact keyword matches (stock codes, numbers) |
| BM25-only | Exact matches, interpretable | No semantic understanding |
| **Hybrid (RRF)** | **Best of both** | **Slightly more complex** |

### Why Not Milvus/Qdrant for 300 PDFs?

- 300 PDFs × ~100 chunks = 30,000 vectors
- FAISS FlatIP is sufficient for this scale
- HNSW only needed if >100k vectors or sub-10ms latency required
- Avoid operational complexity for interview project

## Consequences

### Positive
- Robust to both keyword and semantic queries
- Reranker improves precision
- Parent-child enables context without token waste

### Negative
- More components to maintain
- Reranker adds latency
- Chunking strategy needs tuning

---
Date: 2026-07-16
