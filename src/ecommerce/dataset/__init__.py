"""
E-commerce dataset module.

Submodules:
- sop_builder: Policy/SOP builder for fictional 3C store
- policy_engine: Deterministic policy matching engine
- canonical_cases: Test case generator
- conversation_generator: Synthetic conversation generator
- pipeline: Data processing pipeline

Usage:
    # Build policies
    from src.ecommerce.dataset import SOPBuilder
    builder = SOPBuilder()
    policies = builder.build_fictional_store_sops()
    
    # Generate cases
    from src.ecommerce.dataset import CanonicalCaseGenerator
    generator = CanonicalCaseGenerator()
    cases = generator.generate_all()
    
    # Generate conversations
    from src.ecommerce.dataset import ConversationGenerator
    conv_gen = ConversationGenerator(policies=policies)
    conversations = conv_gen.generate_from_cases(cases)
    
    # Run pipeline
    from src.ecommerce.dataset import run_pipeline
    splits = run_pipeline()
"""

# Import for convenience
from .sop_builder import SOPBuilder, build_sops
from .policy_engine import PolicyEngine, PolicyMatch, Decision, SlotSchema
from .canonical_cases import CanonicalCase, CanonicalCaseGenerator, CaseType, INTENTS as INTENT_LIST  # noqa: F401
from .conversation_generator import ConversationGenerator, SyntheticConversation
from .pipeline import DataPipeline, run_pipeline

__all__ = [
    # SOP Builder
    "SOPBuilder",
    "build_sops",
    # Policy Engine
    "PolicyEngine",
    "PolicyMatch",
    "Decision",
    "SlotSchema",
    # Canonical Cases
    "CanonicalCase",
    "CanonicalCaseGenerator",
    "CaseType",
    "INTENT_LIST",
    # Conversation Generator
    "ConversationGenerator",
    "SyntheticConversation",
    # Pipeline
    "DataPipeline",
    "run_pipeline",
]
