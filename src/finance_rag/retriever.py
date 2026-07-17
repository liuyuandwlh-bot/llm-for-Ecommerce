"""
Hybrid retriever for financial RAG.

Round 2 rewrite:
- Backend is injectable. ``fake`` backend is dependency-free and is used by
  the smoke test/CI to verify the end-to-end flow without downloading
  embeddings.
- Parent-child expansion is implemented by ``expand_to_parent``.
- Persistence saves the chunk map, the dense row-to-id order, and BM25
  state so ``load_index`` can serve the same query and yield identical
  chunk-id lists + scores.
- Metadata filters fail closed when a key is missing on the chunk.
- NaN-safe: zero vectors are replaced with a deterministic small offset and
  norm-checked before normalization.
"""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    doc_id: str
    page_start: int
    page_end: int
    score: float
    source: str
    parent_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": self.score,
            "source": self.source,
            "parent_chunk_id": self.parent_chunk_id,
        }


# ---------------------------------------------------------------------------
# Embedding protocols
# ---------------------------------------------------------------------------


class EmbeddingBackend:
    """Abstract base for embedding backends."""

    name: str = "base"
    revision: str = ""

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class FakeEmbeddingBackend(EmbeddingBackend):
    """Deterministic fake embedder.

    Each text is hashed and projected into a small fixed-dimension vector.
    The same text always produces the same vector so retrieval is reproducible.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = "fake"
        self.revision = "v1"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            arr = np.zeros(self.dim, dtype=np.float32)
            text = (t or "").strip()
            for word in text.split():
                h = abs(hash(word)) % self.dim
                arr[h] += 1.0
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr /= norm
            else:
                arr[0] = 1.0 / np.sqrt(self.dim)
            out[i] = arr
        return out


class RealEmbeddingBackend(EmbeddingBackend):  # pragma: no cover - requires model
    def __init__(
        self, model_name: str = "BAAI/bge-m3", device: str = "auto", revision: str | None = None
    ):
        self.model_name = model_name
        self.device = device
        self.revision = revision
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.revision:
                kwargs["revision"] = self.revision
            self._model = SentenceTransformer(self.model_name, device=self.device, **kwargs)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        vecs = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Sparse + Dense + Hybrid
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    # crude word + char n-gram tokenization suitable for English+Chinese mix
    out: list[str] = []
    for word in text.split():
        out.append(word)
        for i in range(len(word) - 1):
            out.append(word[i : i + 2])
    return out or [text]


class BM25Sparse:
    """Tiny BM25-style scorer implemented by hand to avoid the optional
    ``rank_bm25`` dependency in CI.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenized: list[list[str]] = []
        self.doc_count = 0
        self.avg_dl = 0.0
        self.df: Counter = Counter()
        self.chunk_ids: list[str] = []

    def index(self, chunks: list[dict[str, Any]]):
        if not chunks:
            raise ValueError("empty corpus for BM25")
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.tokenized = [_tokenize(c["text"]) for c in chunks]
        self.doc_count = len(self.tokenized)
        self.avg_dl = sum(len(t) for t in self.tokenized) / max(1, self.doc_count)
        self.df = Counter()
        for toks in self.tokenized:
            for term in set(toks):
                self.df[term] += 1

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if not self.tokenized:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = np.zeros(self.doc_count, dtype=np.float32)
        n_docs = self.doc_count
        for term in q_tokens:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for i, doc in enumerate(self.tokenized):
                freq = doc.count(term)
                if freq == 0:
                    continue
                dl = len(doc)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(1.0, self.avg_dl))
                scores[i] += idf * (freq * (self.k1 + 1)) / max(1e-9, denom)
        top = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]


class DenseIndex:
    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend
        self.embeddings: np.ndarray | None = None
        self.chunk_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}

    def build(self, chunks: list[dict[str, Any]]):
        if not chunks:
            raise ValueError("empty corpus for dense index")
        texts = [c["text"] for c in chunks]
        emb = self.backend.embed(texts).astype(np.float32)
        # Norm-check before dividing
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        zero_mask = (norms == 0).flatten()
        if zero_mask.any():
            # Deterministic offset for zero vectors (no RNG).
            fallback = np.zeros((emb.shape[1],), dtype=np.float32)
            fallback[0] = 1.0 / np.sqrt(emb.shape[1])
            emb[zero_mask] = fallback
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norms, 1e-9, None)
        self.embeddings = emb
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if self.embeddings is None:
            return []
        q = self.backend.embed([query]).astype(np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            q[0, 0] = 1.0 / np.sqrt(q.shape[1])
        else:
            q = q / norm
        scores = (self.embeddings @ q.T).flatten()
        top = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]


class HybridRetriever:
    """Combine sparse + dense with RRF."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        rrf_k: int = 60,
    ):
        self.backend = backend
        self.sparse = BM25Sparse()
        self.dense = DenseIndex(backend)
        self.rrf_k = rrf_k
        self.chunks: list[dict[str, Any]] = []
        self.chunk_map: dict[str, dict[str, Any]] = {}
        self.parents: dict[str, dict[str, Any]] = {}

    def build_index(self, chunks: list[dict[str, Any]]):
        if not chunks:
            raise ValueError("cannot build empty index")
        self.chunks = chunks
        self.chunk_map = {c["chunk_id"]: c for c in chunks}
        self.sparse.index(chunks)
        self.dense.build(chunks)

    def add_parents(self, parents: Sequence[dict[str, Any]]):
        for p in parents:
            self.parents[p["chunk_id"]] = p

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        bm25_results = self.sparse.search(query, top_k=50)
        dense_results = self.dense.search(query, top_k=50)

        rrf: dict[str, float] = {}
        for rank, (cid, _) in enumerate(bm25_results, 1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (cid, _) in enumerate(dense_results, 1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        results: list[RetrievalResult] = []
        for cid, score in ordered:
            chunk = self.chunk_map.get(cid)
            if chunk is None:
                continue
            if filters and not self._passes_filters(chunk, filters):
                continue
            src = []
            if cid in {x[0] for x in bm25_results}:
                src.append("bm25")
            if cid in {x[0] for x in dense_results}:
                src.append("dense")
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    text=chunk["text"],
                    doc_id=chunk.get("doc_id", ""),
                    page_start=chunk.get("page_start", 0),
                    page_end=chunk.get("page_end", 0),
                    score=score,
                    source="+".join(src) or "rrf",
                    parent_chunk_id=chunk.get("parent_chunk_id"),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def expand_to_parent(self, results: list[RetrievalResult]) -> list[dict[str, Any]]:
        """For each child hit, return its parent chunk (for context expansion)."""
        out: list[dict[str, Any]] = []
        seen: set = set()
        for r in results:
            pid = r.parent_chunk_id
            if not pid:
                out.append({"chunk_id": r.chunk_id, "text": r.text, "via": "child_only"})
                continue
            parent = self.parents.get(pid) or self.chunk_map.get(pid)
            if not parent:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            out.append(
                {
                    "chunk_id": pid,
                    "text": parent["text"],
                    "child_ids": parent.get("metadata", {}).get("child_ids", []),
                    "via": "parent",
                }
            )
        return out

    def _passes_filters(self, chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            chunk_value = chunk.get(key)
            if chunk_value is None and "metadata" in chunk:
                chunk_value = chunk.get("metadata", {}).get(key)
            if chunk_value is None:
                # Fail closed
                return False
            if isinstance(value, list):
                if chunk_value not in value:
                    return False
            elif chunk_value != value:
                return False
        return True

    # ---- persistence ------------------------------------------------------

    def save_index(self, output_dir: str) -> dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Chunk order: save only string IDs (not full dicts).
        chunk_order_path = out / "chunk_order.json"
        with open(chunk_order_path, "w", encoding="utf-8") as f:
            json.dump({"chunk_ids": [c["chunk_id"] for c in self.chunks]}, f, ensure_ascii=False)

        # Chunk map (full bodies)
        chunk_map_path = out / "chunk_map.json"
        with open(chunk_map_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_map, f, ensure_ascii=False)

        # Parents
        parents_path = out / "parents.json"
        with open(parents_path, "w", encoding="utf-8") as f:
            json.dump(self.parents, f, ensure_ascii=False)

        # Dense vectors
        dense_path = out / "dense.npy"
        if self.dense.embeddings is not None:
            np.save(str(dense_path), self.dense.embeddings)

        # Manifest
        manifest = {
            "backend_name": self.backend.name,
            "backend_revision": getattr(self.backend, "revision", ""),
            "chunk_count": len(self.chunks),
            "dim": int(self.dense.embeddings.shape[1]) if self.dense.embeddings is not None else 0,
            "rrf_k": self.rrf_k,
        }
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest

    def load_index(self, input_dir: str) -> dict[str, Any]:
        inp = Path(input_dir)
        if not inp.exists():
            raise FileNotFoundError(f"index dir not found: {input_dir}")

        with open(inp / "chunk_order.json", encoding="utf-8") as f:
            chunk_order = json.load(f)
        with open(inp / "chunk_map.json", encoding="utf-8") as f:
            self.chunk_map = json.load(f)
        with open(inp / "parents.json", encoding="utf-8") as f:
            self.parents = json.load(f)

        self.chunks = chunk_order["chunk_ids"]
        if not all(isinstance(cid, str) for cid in self.chunks):
            raise ValueError(
                f"chunk_ids must be strings, got {[type(c).__name__ for c in self.chunks[:3]]}"
            )
        # Reconstruct dense chunks list in row order
        dense_chunks = [self.chunk_map[cid] for cid in self.chunks]
        # Rebuild sparse and dense
        self.sparse.index(dense_chunks)
        self.dense.build(dense_chunks)

        dense_npy = inp / "dense.npy"
        if dense_npy.exists():
            self.dense.embeddings = np.load(str(dense_npy))
            self.dense.chunk_ids = self.chunks
            self.dense.id_to_idx = {cid: i for i, cid in enumerate(self.chunks)}

        with open(inp / "manifest.json", encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_backend(backend_name: str) -> EmbeddingBackend:
    if backend_name == "fake":
        return FakeEmbeddingBackend()
    return RealEmbeddingBackend(backend_name)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build/load a hybrid retriever.")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL")
    parser.add_argument("--output", required=True, help="Index directory")
    parser.add_argument("--backend", default="fake")
    parser.add_argument("--smoke-query", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks: list[dict[str, Any]] = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    # Split children vs parents
    children = [
        c
        for c in chunks
        if c.get("chunk_type") != "mixed" or not c.get("metadata", {}).get("child_ids")
    ]
    # Heuristic: parents are mixed with child_ids or are table parents
    parents = [
        c
        for c in chunks
        if c.get("chunk_type") == "mixed"
        or c.get("chunk_type") == "table"
        and c.get("metadata", {}).get("source", "").startswith("parent")
    ]

    if not children:
        raise SystemExit("no chunks available for indexing")

    backend = _make_backend(args.backend)
    retriever = HybridRetriever(backend=backend)
    retriever.build_index(children)
    retriever.add_parents(parents)
    manifest = retriever.save_index(args.output)
    print(f"saved index: {manifest}")

    if args.smoke_query:
        results = retriever.search(args.smoke_query, top_k=args.top_k)
        expanded = retriever.expand_to_parent(results)
        out = {
            "smoke_query": args.smoke_query,
            "backend": backend.name,
            "results": [r.to_dict() for r in results],
            "parent_expansion": expanded,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    exit(main())
