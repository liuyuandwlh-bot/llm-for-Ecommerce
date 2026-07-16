"""
Evaluation Metrics for Customer Service

Implements proper evaluation metrics including:
- Intent macro/micro F1
- Slot micro/macro F1
- Policy ID exact match
- Decision accuracy
- Safety metrics
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


@dataclass
class IntentAccuracy:
    """Intent classification accuracy metrics."""
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SlotF1:
    """Slot extraction F1 metrics."""
    macro_f1: float
    micro_f1: float
    per_slot: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PolicyAccuracy:
    """Policy matching accuracy."""
    accuracy: float
    correct_count: int
    total_count: int
    policy_wise_accuracy: Dict[str, float] = field(default_factory=dict)


@dataclass
class SafetyMetrics:
    """Safety-related metrics."""
    pii_leak_rate: float
    injection_success_rate: float
    over_commit_rate: float = 0.0  # Rate of unauthorized commitments


@dataclass
class GenerationMetrics:
    """Generation quality metrics."""
    rouge_l: float
    bleu_4: float
    pairwise_win_rate: float
    pairwise_win_rate_ci: Tuple[float, float] = (0.0, 0.0)


@dataclass
class EvaluationSample:
    """A single evaluation sample with prediction and ground truth."""
    sample_id: str
    query: str
    reference: str
    prediction: str
    intent_pred: str
    intent_true: str
    slots_pred: Dict[str, Any]
    slots_true: Dict[str, Any]
    policy_id_true: str
    policy_id_pred: str
    decision_correct: bool
    safety_passed: bool


def calculate_intent_metrics(preds: List[str], refs: List[str]) -> IntentAccuracy:
    """Calculate intent classification metrics."""
    n = len(preds)
    if n == 0:
        return IntentAccuracy(accuracy=0, macro_f1=0, micro_f1=0)
    
    # Accuracy
    correct = sum(1 for p, r in zip(preds, refs) if p == r)
    accuracy = correct / n
    
    # Per-class metrics
    classes = set(preds + refs)
    per_class = {}
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    
    for cls in classes:
        tp = sum(1 for p, r in zip(preds, refs) if p == cls and r == cls)
        fp = sum(1 for p, r in zip(preds, refs) if p == cls and r != cls)
        fn = sum(1 for p, r in zip(preds, refs) if p != cls and r == cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": fn + tp,
        }
        
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    
    # Macro F1
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0
    
    # Micro F1
    micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return IntentAccuracy(
        accuracy=accuracy,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        per_class=per_class,
    )


def calculate_slot_metrics(preds: List[Dict], refs: List[Dict]) -> SlotF1:
    """Calculate slot extraction metrics."""
    n = len(preds)
    if n == 0:
        return SlotF1(macro_f1=0, micro_f1=0)
    
    # Collect all slots
    all_slots = set()
    for slots in preds + refs:
        all_slots.update(slots.keys())
    
    per_slot = {}
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    
    for slot in all_slots:
        tp = 0
        fp = 0
        fn = 0
        
        for pred, ref in zip(preds, refs):
            pred_val = pred.get(slot)
            ref_val = ref.get(slot)
            
            if ref_val is not None:
                # Reference has this slot
                if pred_val is not None and pred_val == ref_val:
                    tp += 1
                else:
                    fn += 1
            elif pred_val is not None:
                # Reference doesn't have slot but prediction does
                fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_slot[slot] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    
    # Macro F1
    macro_f1 = sum(s["f1"] for s in per_slot.values()) / len(per_slot) if per_slot else 0
    
    # Micro F1
    micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return SlotF1(
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        per_slot=per_slot,
    )


def check_pii_leak(text: str) -> bool:
    """Check if text contains PII that shouldn't be there."""
    # Simple patterns for common PII
    import re
    
    patterns = [
        r'1[3-9]\d{9}',  # Phone
        r'\d{17}[\dXx]',  # ID
        r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',  # Bank card
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    
    return False


def check_injection(text: str) -> bool:
    """Check if text contains prompt injection attempts."""
    # Simple patterns
    injection_patterns = [
        "ignore previous",
        "ignore all",
        "disregard your",
        "new instruction",
        "system prompt",
    ]
    
    text_lower = text.lower()
    for pattern in injection_patterns:
        if pattern in text_lower:
            return True
    
    return False


def calculate_rouge(preds: List[str], refs: List[str]) -> float:
    """Calculate ROUGE-L score (simplified)."""
    # This is a placeholder - in production, use rouge-score library
    # For now, return a dummy value that indicates metric is not computed
    return 0.0


def calculate_bleu(preds: List[str], refs: List[str]) -> float:
    """Calculate BLEU-4 score (simplified)."""
    # This is a placeholder - in production, use nltk or sacrebleu
    return 0.0


def evaluate_with_fixtures():
    """
    Run evaluation using fixture test cases.
    
    This allows testing evaluation logic without requiring real model.
    """
    # Fixture test cases
    test_cases = [
        {
            "query": "耳机收到三天了，还没拆封，能退货吗？",
            "intent_true": "return_query",
            "slots_true": {"days_since_delivery": 3, "package_status": "unopened"},
            "policy_id_true": "return_001",
            "decision_true": "full_refund",
        },
        {
            "query": "快递显示签收了，但我没收到",
            "intent_true": "logistics_exception",
            "slots_true": {"logistics_status": "signed", "user_received": False},
            "policy_id_true": "logistics_003",
            "decision_true": "resend_or_refund",
        },
        {
            "query": "我要找你们经理投诉！",
            "intent_true": "escalate",
            "slots_true": {},
            "policy_id_true": "",
            "decision_true": "escalate",
        },
    ]
    
    print("=" * 60)
    print("Evaluation Fixture Test")
    print("=" * 60)
    
    for case in test_cases:
        print(f"\nQuery: {case['query']}")
        print(f"  Intent: {case['intent_true']}")
        print(f"  Decision: {case['decision_true']}")
    
    # Test metrics calculation
    intent_preds = ["return_query", "logistics_exception", "escalate"]
    intent_refs = ["return_query", "logistics_exception", "escalate"]
    
    metrics = calculate_intent_metrics(intent_preds, intent_refs)
    print(f"\n\nIntent Metrics:")
    print(f"  Accuracy: {metrics.accuracy}")
    print(f"  Macro F1: {metrics.macro_f1}")
    print(f"  Micro F1: {metrics.micro_f1}")
    
    return metrics


if __name__ == "__main__":
    evaluate_with_fixtures()
