"""
Near-deduplication via normalized character n-gram Jaccard.

Round 2 design:
- Compare only the business user-utterance text
- Threshold (0.0..1.0) actually controls the result
- Cluster IDs derived from stable SHA-256 over the cluster's content
- Same text but conflicting labels -> quarantine, not silent drop
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from src.common.hashing import Hashing

_DEFAULT_NGRAM = 4


def normalize_text(text: str) -> str:
    """Aggressive normalize to make surface-level differences transparent."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.lower()
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(
        r"[，。！？,.!?；;:：、\u201c\u201d\u2018\u2019\"'()\[\]<>《》「」『』\-_/\\|·•]+", "", s
    )
    return s


def ngrams(text: str, n: int = _DEFAULT_NGRAM) -> set:
    if not text:
        return set()
    s = normalize_text(text)
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def extract_user_business_text(sample: dict) -> str:
    """Pull only the user turns, joined with a stable separator."""
    messages = sample.get("messages") or []
    user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_texts:
        return sample.get("user_text") or sample.get("query") or ""
    return "\n".join(user_texts)


def build_cluster_id(sample_ids: list[str], intent: str) -> str:
    return "nd_" + Hashing.short("cluster", intent, sorted(sample_ids), length=12)


@dataclass
class DedupResult:
    kept_sample_ids: list[str]
    near_removed: list[dict]
    quarantined: list[dict]
    clusters: dict[str, list[str]]
    stats: dict[str, int]

    @property
    def has_label_conflict(self) -> bool:
        return bool(self.quarantined)


def near_dedup(
    samples: list[dict],
    threshold: float = 0.7,
    ngram: int = _DEFAULT_NGRAM,
) -> DedupResult:
    """Group samples by user text n-gram Jaccard similarity.

    Behavior:
    - If two samples have user_text similarity >= threshold, they fall in the
      same cluster.
    - Within each cluster, label/decision disagreement -> quarantine (not drop)
      and remove from "kept".
    - When all members agree, keep the lexicographically smallest sample_id
      (stable across processes) and record the rest as `near_removed` with a
      reason pointing to the kept id.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")

    user_texts: list[str] = [extract_user_business_text(s) for s in samples]
    ngrams_per_sample: list[set] = [ngrams(t, n=ngram) for t in user_texts]

    parent: list[int] = list(range(len(samples)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if find(i) == find(j):
                continue
            sim = jaccard(ngrams_per_sample[i], ngrams_per_sample[j])
            if sim >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(samples)):
        clusters[find(idx)].append(idx)

    kept: list[str] = []
    near_removed: list[dict] = []
    quarantined: list[dict] = []
    cluster_id_map: dict[str, list[str]] = {}

    for cluster_indices in clusters.values():
        ids_in_cluster = [samples[i]["sample_id"] for i in cluster_indices]
        cluster_id = build_cluster_id(
            ids_in_cluster, samples[cluster_indices[0]].get("intent", "unknown")
        )
        cluster_id_map[cluster_id] = ids_in_cluster

        decisions = {samples[i].get("decision") for i in cluster_indices}
        policy_ids = {tuple(sorted(samples[i].get("policy_ids") or [])) for i in cluster_indices}

        for i in cluster_indices:
            samples[i]["dedup_cluster_id"] = cluster_id

        if len(decisions) > 1 or len(policy_ids) > 1:
            # Label conflict -> quarantine all
            for i in cluster_indices:
                quarantined.append(
                    {
                        "sample_id": samples[i]["sample_id"],
                        "reason": "near_dup_label_conflict",
                        "cluster_id": cluster_id,
                    }
                )
            continue

        # Stable canonical member: smallest sample_id
        canonical = min(cluster_indices, key=lambda i: samples[i]["sample_id"])
        kept.append(samples[canonical]["sample_id"])
        for i in cluster_indices:
            if i == canonical:
                continue
            near_removed.append(
                {
                    "sample_id": samples[i]["sample_id"],
                    "kept_sample_id": samples[canonical]["sample_id"],
                    "reason": "near_dup",
                    "cluster_id": cluster_id,
                }
            )

    stats = {
        "input": len(samples),
        "clusters": len(clusters),
        "kept": len(kept),
        "near_removed": len(near_removed),
        "quarantined": len(quarantined),
    }

    return DedupResult(
        kept_sample_ids=kept,
        near_removed=near_removed,
        quarantined=quarantined,
        clusters=cluster_id_map,
        stats=stats,
    )
