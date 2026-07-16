# ADR-001: Model Selection for Customer Service

## Status
Accepted

## Context
We need to select a base model for the e-commerce customer service fine-tuning project. The primary constraints are:
- Single GPU with 24-48GB VRAM (RTX 3090/4090 or A6000)
- Chinese language support
- Commercial-friendly license
- Fast iteration for ablation experiments

## Decision
Use `Qwen/Qwen3-4B-Instruct-2507` as the primary model with `Qwen/Qwen3-8B` as the capacity control group.

### Selection Criteria

| Criterion | Qwen3-4B | Qwen3-8B | Decision |
|-----------|-----------|----------|----------|
| VRAM (FP16) | ~8GB | ~16GB | 4B fits easily |
| VRAM (QLoRA) | ~10GB | ~18GB | 4B leaves room |
| License | Apache-2.0 | Apache-2.0 | Both OK |
| Chinese capability | High | High | Both OK |
| Iteration speed | Fast | Medium | 4B preferred |

## Consequences

### Positive
- Fast experimentation cycle
- Lower computational cost
- Easier to debug and iterate
- Sufficient for customer service task complexity

### Negative
- Smaller model may have lower ceiling
- Need to verify 4B is sufficient for task
- May need to upgrade to 8B if 4B insufficient

## Review
Re-evaluate if:
- 4B model shows systematic failures on complex queries
- Task complexity increases significantly
- More VRAM becomes available

---
Date: 2026-07-16
