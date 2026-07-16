"""
E-commerce dataset module.

Round 2 layout.

Public surface is intentionally narrow: the heavy logic lives in the
submodules; this ``__init__`` only re-exports a small set of convenience
classes/functions used by scripts and tests.
"""

from .canonical_cases import (
    INTENTS,
    CanonicalCase,
    CanonicalCaseGenerator,
    CaseType,
    validate_canonical_cases,
)
from .conversation_generator import (
    ConversationGenerator,
    DeterministicConversationGenerator,
    LLMUnavailableError,
    SyntheticConversation,
)
from .pipeline import (
    STAGES,
    # Backward-compat aliases for tests
    DataPipeline,
    PipelineConfig,
    StageStats,
    check_leakage,
    run_pipeline,
    stratified_group_split,
)
from .policy_engine import BehaviorIntent, Decision, PolicyEngine, PolicyMatch, SlotSchema
from .sop_builder import SOPBuilder, build_sops

# Re-export INTENTS as INTENT_LIST for the original API
INTENT_LIST = INTENTS


__all__ = [
    "SOPBuilder",
    "build_sops",
    "PolicyEngine",
    "PolicyMatch",
    "Decision",
    "SlotSchema",
    "BehaviorIntent",
    "CanonicalCase",
    "CanonicalCaseGenerator",
    "CaseType",
    "INTENTS",
    "INTENT_LIST",
    "validate_canonical_cases",
    "ConversationGenerator",
    "DeterministicConversationGenerator",
    "SyntheticConversation",
    "LLMUnavailableError",
    "PipelineConfig",
    "run_pipeline",
    "stratified_group_split",
    "check_leakage",
    "StageStats",
    "STAGES",
    "DataPipeline",
]
