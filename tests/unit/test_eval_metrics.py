"""Unit tests for evaluation metrics — parse failure handling and accuracy."""

import pytest

from src.ecommerce.eval.metrics import (
    PARSE_FAILURE_SENTINEL,
    f1_metrics,
    summarize_evaluation,
)


class TestParseFailureHandling:
    """§10.1: parse failure must count as error but not crash."""

    def test_mixed_success_and_failure_no_crash(self):
        records = [
            {"parsed": {"intent": "return_query"}, "reference": {"intent": "return_query"}},
            {"parsed": None, "reference": {"intent": "return_query"}},
        ]
        result = summarize_evaluation(records)
        assert "intent" in result
        assert result["parse_failure_rate"] == 0.5
        assert result["parse_failure_count"] == 1

    def test_all_failures_no_crash(self):
        records = [
            {"parsed": None, "reference": {"intent": "return_query"}},
            {"parsed": None, "reference": {"intent": "exchange_query"}},
        ]
        result = summarize_evaluation(records)
        assert result["parse_failure_rate"] == 1.0
        assert result["n"] == 2

    def test_zero_failures(self):
        records = [
            {"parsed": {"intent": "return_query"}, "reference": {"intent": "return_query"}},
            {"parsed": {"intent": "exchange_query"}, "reference": {"intent": "exchange_query"}},
        ]
        result = summarize_evaluation(records)
        assert result["parse_failure_rate"] == 0.0
        assert result["parse_failure_count"] == 0


class TestF1MetricsSentinel:
    """§10.1: None normalized to sentinel so it sorts without TypeError."""

    def test_none_vs_string_no_typeerror(self):
        preds = [None, "return_query", None]
        refs = ["return_query", "return_query", None]
        # Must not raise TypeError
        result = f1_metrics(preds, refs)
        assert "macro_f1" in result

    def test_sentinel_not_in_per_class_keys(self):
        # The sentinel itself should appear in per_class
        preds = [None, "return_query"]
        refs = ["return_query", None]
        result = f1_metrics(preds, refs)
        assert PARSE_FAILURE_SENTINEL in result["per_class"]


class TestMetricAccuracy:
    """§10.2: hand-calculated exact metrics."""

    def test_perfect_match(self):
        records = [
            {"parsed": {"intent": "return_query"}, "reference": {"intent": "return_query"}},
            {"parsed": {"intent": "exchange_query"}, "reference": {"intent": "exchange_query"}},
        ]
        result = summarize_evaluation(records)
        assert result["intent"]["accuracy"] == 1.0

    def test_one_wrong(self):
        records = [
            {"parsed": {"intent": "return_query"}, "reference": {"intent": "return_query"}},
            {"parsed": {"intent": "exchange_query"}, "reference": {"intent": "return_query"}},
        ]
        result = summarize_evaluation(records)
        assert 0.0 < result["intent"]["accuracy"] < 1.0

    def test_decision_wrong(self):
        records = [
            {"parsed": {"decision": "full_refund"}, "reference": {"decision": "full_refund"}},
            {"parsed": {"decision": "reject"}, "reference": {"decision": "full_refund"}},
        ]
        result = summarize_evaluation(records)
        assert result["decision"]["accuracy"] == 0.5

    def test_parse_failure_counts_as_wrong(self):
        # parse failure vs correct reference = 1 wrong
        records = [
            {"parsed": {"intent": "return_query"}, "reference": {"intent": "return_query"}},
            {"parsed": None, "reference": {"intent": "return_query"}},
        ]
        result = summarize_evaluation(records)
        # One correct, one wrong
        assert result["intent"]["accuracy"] == 0.5
        assert result["parse_failure_count"] == 1
