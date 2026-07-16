"""
E-commerce Customer Service Evaluation Module

Based on the recommended project plan:
- 400 gold test cases: normal 180, missing slot 60, edge 60, cross-intent 30, emotional 30, adversarial 20, escalation 20
- Multi-level metrics: intent/slot accuracy, policy accuracy, behavior, safety, generation quality
"""

from .evaluator import CustomerServiceEvaluator, EvaluationResult
from .metrics import (
    IntentAccuracy,
    SlotF1,
    PolicyAccuracy,
    SafetyMetrics,
    GenerationMetrics,
)

__all__ = [
    "CustomerServiceEvaluator",
    "EvaluationResult",
    "IntentAccuracy",
    "SlotF1",
    "PolicyAccuracy",
    "SafetyMetrics",
    "GenerationMetrics",
]
