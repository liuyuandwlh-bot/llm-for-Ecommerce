"""
Hybrid Retriever for Financial RAG

Based on recommended plan:
- BM25 + Dense + Reciprocal Rank Fusion (RRF)
- Metadata filtering (company, year, document type)
- FAISS for dense vectors
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk_id: str
    text: str
    doc_id: str
    page_start: int
    page_end: int
    score: float
    source: str  # bm25, dense, or rrf


@dataclass
class RetrievalQuery:
    """A retrieval query."""
    query_id: str
    text: str
    filters: Dict = None  # Metadata filters


class BM25Retriever:
    """BM25 sparse retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq = {}  # Document frequency for each term
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_count = 0
        self.corpus = []  # List of (chunk_id, text)
        self.tokenized_corpus = []

    def index(self, chunks: List[dict]):
        """Build BM25 index."""
        from rank_bm25 import BM25Okapi

        self.corpus = [(c["chunk_id"], c["text"]) for c in chunks]
        self.tokenized_corpus = [self._tokenize(text) for _, text in self.corpus]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.doc_count = len(self.corpus)

        print(f"BM25 indexed {self.doc_count} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple Chinese tokenization."""
        import jieba
        return list(jieba.cut(text))

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Search BM25 index."""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.corpus[i][0], float(scores[i]))
            for i in top_indices if scores[i] > 0
        ]


class DenseRetriever:
    """Dense vector retriever using sentence transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

        self.index = None
        self.chunks = []  # List of chunk dicts
        self.id_to_idx = {}

    def index(self, chunks: List[dict], batch_size: Optional[int] = None):
        """Build dense index."""
        self.chunks = chunks
        self.id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

        # Encode all chunks
        texts = [c["text"] for c in chunks]
        batch_size = batch_size or self.batch_size

        print(f"Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))

        print(f"Dense index built with {self.index.ntotal} vectors")

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Search dense index."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        return [
            (self.chunks[i]["chunk_id"], float(scores[0][j]))
            for j, i in enumerate(indices[0])
            if i >= 0
        ]


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and Dense.

    Based on recommended plan:
    - BM25 Top50 + Dense Top50
    - Reciprocal Rank Fusion (RRF): score = Σ 1/(k + rank_i)
    - k = 60 (typical value)
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        rrf_k: int = 60,
        device: str = "auto",
    ):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model_name=embedding_model, device=device)
        self.rrf_k = rrf_k

        self.chunks = []  # For text lookup
        self.chunk_map = {}

    def index(self, chunks: List[dict]):
        """Build both BM25 and dense indexes."""
        self.chunks = chunks
        self.chunk_map = {c["chunk_id"]: c for c in chunks}

        # BM25
        print("Building BM25 index...")
        self.bm25.index(chunks)

        # Dense
        print("Building dense index...")
        self.dense.index(chunks)

        print(f"Hybrid index built: {len(chunks)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        Search with hybrid retrieval and RRF.

        1. BM25 Top50
        2. Dense Top50
        3. RRF fusion
        4. Apply metadata filters
        5. Return TopK
        """
        # Get BM25 results
        bm25_results = self.bm25.search(query, top_k=50)
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}

        # Get dense results
        dense_results = self.dense.search(query, top_k=50)
        dense_scores = {chunk_id: score for chunk_id, score in dense_results}

        # RRF fusion
        all_chunk_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        rrf_scores = {}
        for rank, (chunk_id, _) in enumerate(bm25_results, 1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (self.rrf_k + rank)

        for rank, (chunk_id, _) in enumerate(dense_results, 1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (self.rrf_k + rank)

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build results with full chunk info
        results = []
        for chunk_id, rrf_score in sorted_results:
            if chunk_id not in self.chunk_map:
                continue

            chunk = self.chunk_map[chunk_id]

            # Apply filters
            if filters and not self._passes_filters(chunk, filters):
                continue

            # Determine source
            source = []
            if chunk_id in bm25_scores:
                source.append("bm25")
            if chunk_id in dense_scores:
                source.append("dense")

            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=chunk["text"],
                doc_id=chunk["doc_id"],
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
                score=rrf_score,
                source="+".join(source),
            ))

            if len(results) >= top_k:
                break

        return results

    def _passes_filters(self, chunk: dict, filters: Dict) -> bool:
        """Check if chunk passes metadata filters."""
        for key, value in filters.items():
            if key not in chunk.get("metadata", {}):
                if key not in ["doc_id", "page_start", "page_end"]:
                    continue

            chunk_value = chunk.get(key) or chunk.get("metadata", {}).get(key)

            if isinstance(value, list):
                if chunk_value not in value:
                    return False
            elif chunk_value != value:
                return False

        return True

    def save_index(self, output_path: str):
        """Save index to disk."""
        # Save chunk map
        chunk_map_path = Path(output_path).parent / "chunk_map.json"
        with open(chunk_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_map, f, ensure_ascii=False)

        # Save FAISS index
        faiss_path = Path(output_path).parent / "dense.index"
        if self.dense.index is not None:
            faiss.write_index(self.dense.index, str(faiss_path))

        print(f"Index saved to {output_path}")

    def load_index(self, index_path: str, chunk_map_path: str):
        """Load index from disk."""
        # Load chunk map
        with open(chunk_map_path, 'r', encoding='utf-8') as f:
            self.chunk_map = json.load(f)

        self.chunks = list(self.chunk_map.values())

        # Load FAISS index
        faiss_path = Path(chunk_map_path).parent / "dense.index"
        if faiss_path.exists():
            self.dense.index = faiss.read_index(str(faiss_path))

        # Rebuild BM25
        self.bm25.index(self.chunks)

        print(f"Index loaded: {len(self.chunks)} chunks")


def build_retriever(
    chunks_path: str,
    embedding_model: str = "BAAI/bge-m3",
    output_path: Optional[str] = None,
) -> HybridRetriever:
    """Build a hybrid retriever from chunks."""
    # Load chunks
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    # Build retriever
    retriever = HybridRetriever(embedding_model=embedding_model)
    retriever.index(chunks)

    # Save if path provided
    if output_path:
        retriever.save_index(output_path)

    return retriever


if __name__ == "__main__":
    print("Hybrid Retriever Module")
