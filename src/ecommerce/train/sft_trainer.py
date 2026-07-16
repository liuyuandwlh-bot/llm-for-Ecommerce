"""
SFT Training Script using TRL + PEFT

Based on the recommended project plan (推荐项目_全面实施规划.md):
- Model: Qwen3-4B-Instruct-2507
- Method: QLoRA
- Baseline order: Prompt-only → RAG-only → SFT → SFT+RAG → SFT+DPO
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForCompletionLM


@dataclass
class SFTConfig:
    """SFT training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    trust_remote_code: bool = True

    # Quantization (QLoRA)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"  # Will be set after loading model

    # Training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 1536
    packing: bool = False

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Logging
    logging_steps: int = 10
    report_to: str = "tensorboard"

    # Output
    output_dir: str = "output/sft_qlora"
    run_name: Optional[str] = None

    # Hardware
    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42


def format_conversation(example):
    """Format conversation into ChatML format."""
    system_prompt = """你是一个专业、热情的3C电子产品店客服，很乐意帮助用户解决问题。
重要原则：
1. 只依据给出的政策信息回答，不要编造
2. 缺少订单信息时先澄清，不要猜测
3. 不确定时建议转人工
4. 保持礼貌和专业"""

    messages = example.get("messages", [])

    # Build text in ChatML format
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    return {"text": text}


def load_conversation_dataset(data_path: str, max_samples: Optional[int] = None):
    """Load conversation dataset from JSONL."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    if max_samples:
        data = data[:max_samples]

    from datasets import Dataset
    return Dataset.from_list(data).map(format_conversation)


def get_target_modules(model):
    """Get target modules for LoRA based on model architecture."""
    # Common target modules for different model families
    target_modules = set()

    for name, module in model.named_modules():
        # Qwen models
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            parts = name.split(".")
            target_modules.add(parts[-1] if parts else name)

    # Default to common attention modules
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return list(target_modules)


class CustomerServiceSFTTrainer:
    """SFT Trainer for customer service model."""

    def __init__(self, config: SFTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Initialize model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")
        print(f"Revision: {self.config.model_revision}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        from transformers import BitsAndBytesConfig

        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
        )

        # Prepare for kbit training
        if self.config.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Get target modules and setup LoRA
        target_modules = get_target_modules(self.model)
        print(f"Target modules: {target_modules}")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        return self.model, self.tokenizer

    def setup_trainer(self, train_dataset, eval_dataset=None):
        """Setup SFTTrainer."""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            max_steps=-1,
            gradient_checkpointing=self.config.gradient_checkpointing,
            bf16=self.config.bf16,
            fp16=False,
            evaluation_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            seed=self.config.seed,
            remove_unused_columns=False,
        )

        # Data collator for completion modeling
        def response_template(tokenizer):
            return "<|im_start|>assistant\n"

        data_collator = DataCollatorForCompletionLM(
            response_template=response_template(self.tokenizer),
            tokenizer=self.tokenizer,
            mlm=False,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            data_collator=data_collator,
        )

        return self.trainer

    def train(self, train_data_path: str, eval_data_path: str = None):
        """Run training."""
        print("Setting up model...")
        self.setup_model()

        print(f"Loading training data from: {train_data_path}")
        train_dataset = load_conversation_dataset(train_data_path)

        eval_dataset = None
        if eval_data_path:
            print(f"Loading evaluation data from: {eval_data_path}")
            eval_dataset = load_conversation_dataset(eval_data_path)

        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Evaluation samples: {len(eval_dataset)}")

        print("Setting up trainer...")
        self.setup_trainer(train_dataset, eval_dataset)

        print("Starting training...")
        self.trainer.train()

        print("Saving final model...")
        self.trainer.save_model()
        self.trainer.save_state()

        return self.trainer


def load_config_from_yaml(yaml_path: str) -> SFTConfig:
    """Load configuration from YAML file."""
    import yaml

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load()

    return SFTConfig(**config_dict)


def train_sft(config_path: str = None, train_data: str = None, eval_data: str = None):
    """Main training function."""
    # Load config
    if config_path:
        config = load_config_from_yaml(config_path)
    else:
        config = SFTConfig()

    # Override with command line args
    if train_data:
        config.output_dir = "output/sft_qlora"

    # Setup training
    trainer = CustomerServiceSFTTrainer(config)

    # Load data
    train_path = train_data or "data/processed/synthetic_conversations.jsonl"
    eval_path = eval_data

    # Run training
    trainer.train(train_path, eval_path)

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--train_data", type=str, help="Training data path")
    parser.add_argument("--eval_data", type=str, help="Evaluation data path")
    args = parser.parse_args()

    train_sft(
        config_path=args.config,
        train_data=args.train_data,
        eval_data=args.eval_data,
    )
