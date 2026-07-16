# LLM Portfolio Platform

> E-commerce Customer Service Fine-tuning + Financial RAG System

## Project Status

**Framework**: Implemented and tested  
**Training Data**: Fixture/demo only (synthetic)  
**Trained Models**: Planned but not run (requires GPU)  
**Public Data**: Planned but not acquired

## Project Overview

This portfolio demonstrates end-to-end LLM system development skills through two complementary projects:

### Project 1: E-commerce Customer Service Fine-tuning
- **Status**: Framework implemented, training pending
- **Data**: Self-generated fictional 3C electronics store policies
- **Intents**: logistics_query, logistics_exception, return_query, exchange_query, specification_query, coupon_or_price_protection, complaint, escalate, tool_required, out_of_scope
- **Models**: Qwen3-8B + Llama-3.1-8B-Instruct (planned)
- **Method**: BF16 LoRA SFT

### Project 2: Financial Report RAG System
- **Status**: Framework implemented, documents pending
- **Data Sources**: Official disclosures only (巨潮资讯, 上交所, 深交所, 央行, 发改委)
- **Retrieval**: BM25 + Dense + RRF + Reranker

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Gateway                          │
├─────────────────────────────────────────────────────────────────┤
│  Route by domain                                                │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ E-commerce       │    │ Financial RAG    │                   │
│  │ Customer Service │    │                  │                   │
│  ├──────────────────┤    ├──────────────────┤                   │
│  │ Intent/Slot      │    │ Query Parser     │                   │
│  │ Recognition      │    │ Metadata Filter  │                   │
│  │                  │    │ BM25 + Dense     │                   │
│  │ Policy Match     │    │ RRF + Reranker  │                   │
│  │                  │    │ Calculator       │                   │
│  ├──────────────────┤    ├──────────────────┤                   │
│  │ vLLM + LoRA      │    │ vLLM (Base)     │                   │
│  │ Adapter          │    │ Citation Engine  │                   │
│  └──────────────────┘    └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
llm-portfolio-platform/
├── configs/              # YAML configuration files
│   └── train/           # SFT configs (Qwen3-8B, Llama-3.1-8B)
├── data/                # Data (see DATA_POLICY.md)
│   ├── fixtures/        # Fixture/demo data
│   ├── registry/        # Data source inventory
│   └── processed/       # Processed outputs
├── src/                 # Source code
│   ├── common/          # Shared utilities (PII, schemas, logging)
│   ├── ecommerce/       # Customer service module
│   │   ├── dataset/     # SOP builder, policy engine, cases, generator, pipeline
│   │   ├── train/      # SFT trainer with Qwen/Llama support
│   │   └── eval/       # Evaluation metrics and evaluator
│   ├── finance_rag/     # Financial RAG module
│   │   ├── pdf_parser/ # Multi-backend PDF parsing
│   │   ├── chunker/    # Structural-aware chunking
│   │   ├── retriever/  # Hybrid retrieval
│   │   └── reranker/   # Cross-encoder reranking
│   └── serving/        # API gateway & deployment
├── tests/               # Unit & integration tests
└── docs/               # Architecture decisions
```

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training/inference)
- 24GB+ VRAM (RTX 3090/4090/A6000)

### Installation

```bash
# Clone repository
git clone https://github.com/liuyuandwlh-bot/llm-for-Ecommerce.git
cd llm-for-Ecommerce

# Install dependencies
pip install -e ".[dev]"  # Or: pip install -e ".[train,rag,dev]"

# Verify installation
pytest -q
```

### Generate Training Data

```bash
# Step 1: Build SOP policies
python -m src.ecommerce.dataset.sop_builder \
  --output data/processed/fixtures/policies.json

# Step 2: Generate canonical cases
python -m src.ecommerce.dataset.canonical_cases \
  --output data/fixtures/ecommerce/canonical_cases.jsonl

# Step 3: Generate synthetic conversations
python -m src.ecommerce.dataset.conversation_generator \
  --policies data/processed/fixtures/policies.json \
  --cases data/fixtures/ecommerce/canonical_cases.jsonl \
  --output data/processed/fixtures/conversations.jsonl \
  --seed 42

# Step 4: Run data pipeline (validation, PII masking, dedup, split)
python -m src.ecommerce.dataset.pipeline \
  --input data/processed/fixtures/conversations.jsonl \
  --output data/processed/fixtures/release_v1 \
  --policies data/processed/fixtures/policies.json \
  --seed 42
```

### Run Tests

```bash
# All tests
pytest -q

# Specific test files
pytest tests/unit/test_ecommerce_dataset.py -v
pytest tests/unit/test_common.py -v
pytest tests/integration/test_api.py -v

# Smoke tests (no GPU required)
python -m src.ecommerce.dataset.sop_builder --help
python -m src.ecommerce.dataset.canonical_cases --help
python -m src.ecommerce.dataset.conversation_generator --help
python -c "from src.serving.gateway.main import app; print(app.title)"
```

### Training (Requires GPU)

```bash
# Train Qwen3-8B
python -m src.ecommerce.train.sft_trainer \
  --config configs/train/sft_qwen3_8b.yaml \
  --train_data data/processed/fixtures/release_v1/train.jsonl \
  --eval_data data/processed/fixtures/release_v1/dev.jsonl

# Train Llama-3.1-8B
python -m src.ecommerce.train.sft_trainer \
  --config configs/train/sft_llama3_8b.yaml \
  --train_data data/processed/fixtures/release_v1/train.jsonl \
  --eval_data data/processed/fixtures/release_v1/dev.jsonl

# Dry run (no GPU required)
python -m src.ecommerce.train.sft_trainer --dry_run --config configs/train/sft_qwen3_8b.yaml
```

### Evaluation

```bash
# Evaluate (mock mode without model)
python -m src.ecommerce.eval.evaluator \
  --test_data data/fixtures/ecommerce/canonical_cases.jsonl

# Evaluate with trained model
python -m src.ecommerce.eval.evaluator \
  --model_path output/sft/qwen3_8b/checkpoint-xxx \
  --test_data data/processed/fixtures/release_v1/test.jsonl \
  --output reports/evaluation.json
```

### API Server

```bash
# Start server
uvicorn src.serving.gateway.main:app --host 0.0.0.0 --port 8000

# Or run module
python -m src.serving.gateway.main
```

## Key Technologies

| Component | Technology |
|-----------|------------|
| Base Model | Qwen3-8B, Llama-3.1-8B-Instruct |
| Fine-tuning | BF16 LoRA (TRL + PEFT) |
| PDF Parsing | pdfplumber, PyMuPDF |
| Embedding | BAAI/bge-m3 |
| Vector Index | FAISS |
| Inference | vLLM |
| API | FastAPI |

## Data Sources

| Source | Status | License | Usage |
|--------|--------|---------|-------|
| Owned SOP | ✅ Acquired | CC0 | Intent/slot, SFT, eval |
| CrossWOZ | ⏳ Planned | Apache-2.0 | Taxonomy reference only |
| MASSIVE | ⏳ Planned | CC BY 4.0 | Intent structure only |

## Model Selection ADR

See [docs/ADR-001-model-selection.md](docs/ADR-001-model-selection.md) for rationale on Qwen3-8B vs Llama-3.1-8B selection.

## License

MIT License - See LICENSE file for details.

## Data Policy

See [DATA_POLICY.md](DATA_POLICY.md) for data governance guidelines.

---

**Note**: All training metrics, accuracy numbers, and model performance results shown in this repository are targets/goals, not actual results. Real training has not been performed yet as it requires GPU resources.
