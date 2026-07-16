"""E-commerce module.

Submodules:
- dataset: SOP, canonical cases, conversation generation, data pipeline
- train: SFT, DPO training
- eval: Evaluation metrics and evaluator
"""

from .dataset import (  # noqa: F401  (re-exports for legacy import paths)
    conversation_generator,
    sop_builder,
)

__all__ = [
    "conversation_generator",
    "sop_builder",
]  # type: ignore[var-annotated]
