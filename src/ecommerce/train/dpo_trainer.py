"""
DPO Training Script for Preference Alignment

Based on the recommended project plan:
- DPO is optional, only if SFT+RAG has preference issues
- Requires 1000-2000 high-quality preference pairs
- Start with 300 pairs to validate signal
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset, load_dataset
from trl import DPOTrainer, DPOConfig


@dataclass
class DPOConfig:
    """DPO training configuration."""
    # Model
    sft_model_path: str = "output/sft_qlora/checkpoint-xxx"
    ref_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"

    # LoRA (for reference model)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # DPO Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    beta: float = 0.1  # DPO temperature parameter
    gamma: float = 1.0  # Label smoothing

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100

    # Output
    output_dir: str = "output/dpo"
    run_name: Optional[str] = None

    # Hardware
    bf16: bool = True
    seed: int = 42


def format_dpo_data(example):
    """Format preference data for DPO."""
    return {
        "prompt": example.get("prompt", ""),
        "chosen": example.get("chosen", ""),
        "rejected": example.get("rejected", ""),
    }


def prepare_dpo_data(input_path: str, output_path: str, num_pairs: int = 300):
    """
    Prepare DPO preference data from conversation dataset.

    In practice, this would involve:
    1. Generate multiple responses per prompt (vary temperature, decoding params)
    2. Human/rule-based labeling of preference
    3. Quality filtering

    For this template, we create synthetic preference pairs.
    """
    print(f"Preparing DPO data from: {input_path}")
    print(f"Target pairs: {num_pairs}")

    # Load conversations
    conversations = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line))

    dpo_data = []

    for conv in conversations[:num_pairs]:
        messages = conv.get("messages", [])

        # Extract user prompt
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            continue

        prompt = user_messages[0].get("content", "")

        # Get assistant response
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_messages:
            continue

        chosen = assistant_messages[0].get("content", "")

        # Create synthetic rejected response (slightly worse)
        rejected = create_worse_response(chosen, conv.get("intent", ""))

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "intent": conv.get("intent", ""),
        })

    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created {len(dpo_data)} DPO pairs")
    return dpo_data


def create_worse_response(chosen: str, intent: str) -> str:
    """Create a worse version of the response for rejected."""
    worse_templates = {
        "return_query": f"{chosen}\n\n【提示】如有其他问题，欢迎随时咨询～",
        "exchange_query": f"好的，我帮您看看。\n{chosen}",
        "logistics_query": f"{chosen}\n\n物流信息仅供参考，以实际为准。",
        "complaint": f"非常抱歉给您带来困扰。{chosen}",
    }

    template = worse_templates.get(intent, chosen)

    # Add some common degradation patterns
    worse_patterns = [
        # Too short
        "好的。",
        # Too mechanical
        "根据您的问题，请查阅相关规定。如有疑问请联系客服。",
        # Missing empathy
        "处理完成。",
        # Incomplete
        chosen[:len(chosen)//2] + "...",
    ]

    import random
    return random.choice(worse_patterns)


class CustomerServiceDPOTrainer:
    """DPO Trainer for preference alignment."""

    def __init__(self, config: DPOConfig):
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Initialize models."""
        print(f"Loading SFT model from: {self.config.sft_model_path}")
        print(f"Reference model: {self.config.ref_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.ref_model_name,
            revision=self.config.model_revision,
            trust_remote_code=True,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load SFT model (with LoRA)
        self.model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                self.config.ref_model_name,
                revision=self.config.model_revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ),
            self.config.sft_model_path,
        )

        # Load reference model (SFT model before DPO training)
        self.ref_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                self.config.ref_model_name,
                revision=self.config.model_revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ),
            self.config.sft_model_path,
        )
        self.ref_model.eval()
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        return self.model, self.ref_model, self.tokenizer

    def setup_trainer(self, train_dataset):
        """Setup DPOTrainer."""
        dpo_args = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            beta=self.config.beta,
            gamma=self.config.gamma,
            bf16=self.config.bf16,
            evaluation_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            logging_steps=10,
            seed=self.config.seed,
        )

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        return self.trainer

    def train(self, train_data_path: str, eval_data_path: str = None):
        """Run DPO training."""
        print("Setting up models...")
        self.setup_model()

        # Load and format data
        print(f"Loading DPO data from: {train_data_path}")
        dataset = load_dataset("json", data_files=train_data_path, split="train")
        dataset = dataset.map(format_dpo_data)

        eval_dataset = None
        if eval_data_path:
            print(f"Loading evaluation data from: {eval_data_path}")
            eval_dataset = load_dataset("json", data_files=eval_data_path, split="train")
            eval_dataset = eval_dataset.map(format_dpo_data)

        print(f"Training pairs: {len(dataset)}")
        if eval_dataset:
            print(f"Evaluation pairs: {len(eval_dataset)}")

        print("Setting up trainer...")
        self.setup_trainer(dataset)

        print("Starting DPO training...")
        self.trainer.train()

        print("Saving final model...")
        self.trainer.save_model()
        self.trainer.save_state()

        return self.trainer


def train_dpo(config_path: str = None, train_data: str = None, eval_data: str = None):
    """Main DPO training function."""
    from .sft_trainer import load_config_from_yaml

    if config_path:
        config = load_config_from_yaml(config_path)
    else:
        config = DPOConfig()

    trainer = CustomerServiceDPOTrainer(config)
    trainer.train(train_data, eval_data)

    print("DPO training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--train_data", type=str, required=True, help="DPO training data path")
    parser.add_argument("--eval_data", type=str, help="DPO evaluation data path")
    args = parser.parse_args()

    train_dpo(
        config_path=args.config,
        train_data=args.train_data,
        eval_data=args.eval_data,
    )
