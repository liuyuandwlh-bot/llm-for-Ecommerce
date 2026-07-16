# LLM Portfolio Platform - Makefile

.PHONY: help install train eval test clean lint format

# Default target
help:
	@echo "LLM Portfolio Platform - Available targets:"
	@echo ""
	@echo "  make install          Install dependencies"
	@echo "  make train-sft       Run SFT training"
	@echo "  make eval            Run evaluation"
	@echo "  make serve           Start API server"
	@echo "  make rag-index       Build RAG index"
	@echo "  make test            Run tests"
	@echo "  make lint            Run linters"
	@echo "  make format          Format code"
	@echo "  make clean           Clean artifacts"

# Installation
install:
	pip install -r requirements.txt
	@echo "Dependencies installed. Run 'source .venv/bin/activate' to activate."

# Data generation
data-sop:
	python -m src.ecommerce.dataset.sop_builder

data-conversations:
	python -m src.ecommerce.dataset.conversation_generator

data-all: data-sop data-conversations

# Training
train-sft:
	python -m src.ecommerce.train.sft_trainer \
		--config configs/train/sft_qlora_4b.yaml \
		--train_data data/processed/synthetic_conversations.jsonl

train-dpo:
	python -m src.ecommerce.train.dpo_trainer \
		--train_data data/processed/dpo_pairs.jsonl

# RAG Pipeline
rag-parse:
	python -m src.finance_rag.pdf_parser \
		data/raw/reports \
		data/interim/parsed

rag-chunk:
	python -m src.finance_rag.chunker \
		data/interim/parsed \
		data/processed/chunks.jsonl

rag-index:
	python -c "from src.finance_rag.retriever import build_retriever; build_retriever('data/processed/chunks.jsonl', output_path='data/interim/index')"

rag-all: rag-parse rag-chunk rag-index

# Evaluation
eval-customer-service:
	python -m src.ecommerce.eval.evaluator \
		--model_path output/sft_qlora/checkpoint-xxx \
		--test_data evaluation/ecommerce_gold/test.jsonl \
		--output reports/evaluation/customer_service.json

eval-all: eval-customer-service

# Serving
serve:
	uvicorn src.serving.gateway.main:app --host 0.0.0.0 --port 8000 --reload

serve-vllm:
	bash scripts/serve_vllm.sh

# Testing
test:
	pytest tests/ -v --cov=src

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Linting
lint:
	ruff check src/

format:
	black src/
	isort src/

# Cleaning
clean:
	rm -rf output/
	rm -rf data/interim/
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker build -t llm-portfolio:latest .

docker-run:
	docker run -p 8000:8000 --gpus all llm-portfolio:latest
