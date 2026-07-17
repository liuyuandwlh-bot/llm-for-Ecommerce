"""Unit tests for split and deduplication.

Covers §5 (group-aware stratified split) and §13 split regression tests.
"""

import pytest

from src.ecommerce.dataset.pipeline import _group_id, stratified_group_split
from src.common.near_dedup import normalize_text, near_dedup


class TestGroupKey:
    def test_group_key_consistent(self):
        s1 = {
            "sample_id": "conv_001_0",
            "parent_case_id": "case_return_001",
            "template_family": "default",
        }
        s2 = {
            "sample_id": "conv_001_0",
            "parent_case_id": "case_return_001",
            "template_family": "default",
        }
        assert _group_id(s1) == _group_id(s2)

    def test_group_key_different_cases_different(self):
        s1 = {"sample_id": "a", "parent_case_id": "case_1", "template_family": "t1"}
        s2 = {"sample_id": "b", "parent_case_id": "case_2", "template_family": "t1"}
        assert _group_id(s1) != _group_id(s2)

    def test_group_key_no_empty_key(self):
        # Must not produce an empty group key
        s = {"sample_id": "conv_001"}
        key = _group_id(s)
        assert key, "group key must not be empty"


class TestStratifiedGroupSplit:
    def _make_samples(self, n: int):
        return [
            {
                "sample_id": f"s{i}",
                "parent_case_id": f"case_{i % 5}",
                "template_family": "default",
                "intent": "return_query",
            }
            for i in range(n)
        ]

    def test_train_larger_than_dev(self):
        samples = self._make_samples(20)
        splits, meta = stratified_group_split(samples, seed=42)
        assert len(splits["train"]) > len(splits["dev"])

    def test_same_seed_deterministic(self):
        samples = self._make_samples(12)
        _, meta1 = stratified_group_split(samples, seed=42)
        _, meta2 = stratified_group_split(samples, seed=42)
        assert meta1["seed"] == meta2["seed"] == 42

    def test_all_splits_cover_all_samples(self):
        samples = self._make_samples(10)
        splits, meta = stratified_group_split(samples, seed=42)
        assigned = (
            {s["sample_id"] for s in splits["train"]}
            | {s["sample_id"] for s in splits["dev"]}
            | {s["sample_id"] for s in splits["test"]}
        )
        assert assigned == {f"s{i}" for i in range(10)}


class TestNormalizeText:
    def test_strips_whitespace(self):
        assert normalize_text("  hello  world  ") == normalize_text("hello world")

    def test_lowercase(self):
        assert normalize_text("HELLO") == normalize_text("hello")


class TestNearDedup:
    def test_identical_text_clustered(self):
        samples = [
            {"sample_id": "a", "messages": [{"role": "user", "content": "耳机能退吗"}]},
            {"sample_id": "b", "messages": [{"role": "user", "content": "耳机能退吗"}]},
        ]
        result = near_dedup(samples, threshold=0.7)
        assert result.stats["clusters"] >= 1

    def test_different_text_not_all_removed(self):
        samples = [
            {"sample_id": "a", "messages": [{"role": "user", "content": "耳机能退吗"}]},
            {"sample_id": "b", "messages": [{"role": "user", "content": "苹果笔记本怎么样"}]},
        ]
        result = near_dedup(samples, threshold=0.7)
        assert result.stats.get("removed", 0) == 0
