"""
E-commerce Module

Submodules:
- dataset: SOP, canonical cases, conversation generation, data pipeline
- train: SFT, DPO training
- eval: Evaluation metrics and evaluator
"""

from .dataset import sop_builder, conversation_generator
