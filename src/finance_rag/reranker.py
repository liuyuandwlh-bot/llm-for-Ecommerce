"""
Cross-Encoder Reranker for Financial RAG

Based on recommended plan:
- Qwen3-Reranker-0.6B or BGE-reranker-v2-m3
- Re-rank top-k from hybrid retriever
- GPU/CPU tradeoff consideration
"""

from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    """A reranked result."""

    chunk_id: str
    text: str
    doc_id: str
    page_start: int
    page_end: int
    original_score: float
    rerank_score: float
    source: str


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    Models:
    - Qwen3-Reranker-0.6B (Apache-2.0)
    - BGE-reranker-v2-m3 (MIT)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = (
            device if device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length
        self.batch_size = batch_size

        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=self.device,
        )

    def rerank(
        self,
        query: str,
        results: list[dict],  # Results from retriever
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        Re-rank retrieval results.

        Args:
            query: The search query
            results: List of retrieval results (dict with chunk_id, text, etc.)
            top_k: Number of results to return

        Returns:
            List of reranked results
        """
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, result["text"]) for result in results]

        # Score in batches
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            batch_scores = self.model.predict(batch_pairs, show_progress_bar=False)
            scores.extend(batch_scores.tolist())

        # Combine with original scores
        reranked = []
        for result, score in zip(results, scores, strict=False):
            reranked.append(
                RerankResult(
                    chunk_id=result["chunk_id"],
                    text=result["text"],
                    doc_id=result.get("doc_id", ""),
                    page_start=result.get("page_start", 0),
                    page_end=result.get("page_end", 0),
                    original_score=result.get("score", 0),
                    rerank_score=float(score),
                    source=result.get("source", ""),
                )
            )

        # Sort by rerank score
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)

        return reranked[:top_k]


def create_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> CrossEncoderReranker:
    """Create a reranker instance."""
    return CrossEncoderReranker(model_name=model_name)


if __name__ == "__main__":
    print("Cross-Encoder Reranker Module")
