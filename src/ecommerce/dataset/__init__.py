"""
E-commerce Customer Service Dataset Module

This module handles:
- Fictional SOP (Standard Operating Procedures) data
- Policy decision tables
- Canonical case generation
- Synthetic conversation generation
"""

from .sop_builder import SOPBuilder, Policy, PolicyDecision
from .intent_classifier import IntentClassifier, Intent, Slot
from .conversation_generator import ConversationGenerator

__all__ = [
    "SOPBuilder",
    "Policy",
    "PolicyDecision",
    "IntentClassifier",
    "Intent",
    "Slot",
    "ConversationGenerator",
]
