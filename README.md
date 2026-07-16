# LLM Portfolio Platform

> E-commerce Customer Service Fine-tuning + Financial RAG System

## Project Overview

This portfolio demonstrates end-to-end LLM system development skills through two complementary projects:

### Project 1: E-commerce Customer Service Fine-tuning
- Fictional 3C electronics store (headphones, chargers, cables)
- 6 intent categories: logistics, returns, exchanges, specifications, coupons, complaints
- QLoRA SFT + optional DPO alignment
- Multi-level evaluation with gold test set

### Project 2: Financial Report RAG System
- Official disclosures (annual reports, policy documents)
- Hybrid retrieval: BM25 + Dense + RRF + Reranker
- Page-level citation with bbox traceability
- Calculator for numerical reasoning

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
│  │ Policy RAG       │    │ RRF + Reranker  │                   │
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
│   ├── train/           # SFT/DPO training configs
│   ├── rag/             # RAG pipeline configs
│   ├── serve/           # Deployment configs
│   └── eval/            # Evaluation configs
├── data/                # Data (see DATA_POLICY.md)
│   ├── registry/        # Data source inventory
│   ├── raw/             # Immutable original data
│   ├── interim/         # Intermediate processing
│   ├── processed/       # Ready for training/RAG
│   ├── splits/          # Frozen train/dev/test
│   └── quarantine/      # Suspicious data pending review
├── src/                 # Source code
│   ├── common/          # Shared utilities
│   ├── ecommerce/       # Customer service module
│   ├── finance_rag/     # Financial RAG module
│   └── serving/         # API gateway & deployment
├── tests/               # Unit & integration tests
├── evaluation/         # Gold sets & qrels
├── reports/             # Experiment reports
├── model_cards/         # Model documentation
└── docs/                # Architecture decisions
```

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training/inference)
- 24GB+ VRAM (RTX 3090/4090/A6000)

### Installation

```bash
# Clone repository
git clone https://github.com/liuyuandwlh-bot/llm-portfolio-platform.git
cd llm-portfolio-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Project 1: Customer Service Training

```bash
# Step 1: Create fictional SOP data
python -m src.ecommerce.dataset.build_sop

# Step 2: Generate synthetic conversations
python -m src.ecommerce.dataset.generate_conversations

# Step 3: Run SFT training
python -m src.ecommerce.train.sft --config configs/train/sft_qlora_4b.yaml

# Step 4: Evaluate
python -m src.ecommerce.eval.evaluate --checkpoint output/sft_qlora/checkpoint-xxx
```

### Project 2: Financial RAG Setup

```bash
# Step 1: Download sample documents (mock for now)
python -m src.finance_rag.ingest.download_samples

# Step 2: Parse PDFs
python -m src.finance_rag.parse.batch_parse --input data/raw/reports

# Step 3: Build index
python -m src.finance_rag.chunk.build_chunks
python -m src.finance_rag.retrieve.build_index

# Step 4: Run RAG pipeline
python -m src.finance_rag.answer.demo
```

### Deployment

```bash
# Start vLLM server
bash scripts/serve_vllm.sh

# Start API gateway
uvicorn src.serving.gateway.main:app --reload
```

## Timeline (13 Weeks)

| Week | Focus | Deliverable |
|------|-------|-------------|
| W0 | Infrastructure | Dev environment, DVC setup, data registry |
| W1 | SOP Data | 50 policies, 200 canonical cases |
| W2 | Conversation Data | 800 conversations, gold test set |
| W3 | SFT Baseline | Prompt/RAG baselines, first training run |
| W4 | Ablation | Rank/data/rank experiments, DPO decision |
| W5 | Customer Service Wrap-up | Final model, bad cases, model card |
| W6 | Financial Data | 120 documents, manifest, parse gold |
| W7 | PDF Parsing | Docling/pdftplumber comparison, chunking |
| W8 | Retrieval | BM25/Dense/RRF, FAISS index |
| W9 | Reranking & Generation | Cross-encoder, citation engine |
| W10 | RAG Evaluation | 300 qrels, full pipeline report |
| W11 | Deployment | vLLM serving, LoRA adapter loading |
| W12 | Performance | Benchmark, cache, fault tolerance |
| W13 | Final Review | Documentation, demo, presentation |

## Key Technologies

| Component | Technology |
|-----------|-------------|
| Base Model | Qwen3-4B-Instruct / Qwen3-8B |
| Fine-tuning | QLoRA (TRL + PEFT) |
| Preference Alignment | DPO (optional) |
| PDF Parsing | pdfplumber + PyMuPDF + PaddleOCR |
| Embedding | Qwen3-Embedding-0.6B / BGE-M3 |
| Reranker | Qwen3-Reranker-0.6B / BGE-reranker-v2-m3 |
| Vector Index | FAISS (FlatIP / HNSW) |
| Inference | vLLM |
| API | FastAPI |
| Tracking | MLflow + DVC |

## License

MIT License - See LICENSE file for details.

## Data Policy

See [DATA_POLICY.md](DATA_POLICY.md) for data governance guidelines.

All synthetic data and fictional policies are created for this project. All real data comes from official government/public disclosures with proper attribution.
