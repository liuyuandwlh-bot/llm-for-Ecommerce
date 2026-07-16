# vLLM Service Script for Customer Service

# Based on recommended plan:
# - Model: Qwen3-4B-Instruct-2507
# - LoRA adapter for customer service
# - GPU memory utilization: 0.85

#!/bin/bash

# Configuration
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION="cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER_PATH="./output/sft_qlora/checkpoint-xxx"
PORT=8000
HOST="0.0.0.0"
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=4096

# Check for LoRA adapter
if [ -d "$ADAPTER_PATH" ]; then
    echo "Using LoRA adapter: $ADAPTER_PATH"
    LORA_ARGS="--enable-lora --lora-modules ecom=$ADAPTER_PATH"
else
    echo "No LoRA adapter found, using base model"
    LORA_ARGS=""
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi

# Start vLLM server
echo "Starting vLLM server..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "GPU Memory: ${GPU_MEMORY_UTILIZATION}%"

vllm serve "$MODEL_NAME" \
    --revision "$MODEL_REVISION" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    $LORA_ARGS \
    --tensor-parallel-size 1
