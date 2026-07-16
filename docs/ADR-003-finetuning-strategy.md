# ADR-003: Fine-tuning Strategy

## Status
Accepted

## Context
Need to determine the fine-tuning approach for e-commerce customer service. Considerations:
- Limited data (estimated 2-3k high-quality samples)
- Need to preserve base model capabilities
- Want to demonstrate understanding of PEFT methods
- Resource constraints (single GPU)

## Decision
Follow this order:
1. **Prompt-only baseline**: Base model + SOP prompt
2. **RAG-only baseline**: Base model + retrieved policies
3. **QLoRA SFT**: Main training method
4. **DPO (optional)**: Only if SFT shows preference issues

### Why QLoRA over Full Fine-tune?

| Method | VRAM | Data Efficiency | Catastrophic Forgetting | Explainability |
|--------|------|-----------------|------------------------|----------------|
| Full FT | 24GB+ | High | High risk | Low |
| LoRA | 18GB | Medium | Medium risk | Medium |
| **QLoRA** | **10-12GB** | **Medium** | **Low** | **Medium** |

### Why Not Start with DPO?

DPO optimizes for preferences (politeness, structure) but:
- Cannot fix factual errors (need correct data/RAG)
- Requires additional annotation effort
- Adds complexity without guaranteed improvement

DPO is added only if SFT+RAG achieves factual correctness but lacks preference quality.

## Consequences

### Positive
- Resource efficient
- Clear ablation path
- Demonstrates systematic approach

### Negative
- May need multiple training runs for ablation
- QLoRA slightly slower than LoRA

---
Date: 2026-07-16
