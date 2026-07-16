"""
SFT Training Script using TRL + PEFT

Supports both Qwen and Llama models with proper chat template handling.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import DataCollatorForCompletionLM

from .sft_config import SFTConfig, MODEL_MATRIX


logger = logging.getLogger(__name__)


def format_conversation_qwen(messages: List[Dict], tokenizer) -> str:
    """Format conversation using Qwen chat template."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def format_conversation_llama(messages: List[Dict], tokenizer) -> str:
    """Format conversation using Llama chat template."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def format_conversation(
    example: Dict, 
    tokenizer,
    chat_template: str = "qwen"
) -> Dict:
    """Format conversation into model-specific format."""
    messages = example.get("messages", [])
    
    if chat_template == "llama":
        text = format_conversation_llama(messages, tokenizer)
    else:
        text = format_conversation_qwen(messages, tokenizer)
    
    return {"text": text}


def load_conversation_dataset(
    data_path: str, 
    tokenizer,
    chat_template: str = "qwen",
    max_samples: Optional[int] = None
) -> Dataset:
    """Load conversation dataset from JSONL and format."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    if max_samples:
        data = data[:max_samples]
    
    dataset = Dataset.from_list(data)
    
    # Format with appropriate template
    def format_fn(example):
        return format_conversation(example, tokenizer, chat_template)
    
    return dataset.map(format_fn, remove_columns=dataset.column_names)


def get_target_modules(model, chat_template: str = "qwen") -> List[str]:
    """Get target modules for LoRA based on model architecture."""
    target_modules = set()
    
    for name, module in model.named_modules():
        # Qwen models
        if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
            parts = name.split(".")
            target_modules.add(parts[-1] if parts else name)
    
    # Llama models have different module names
    if chat_template == "llama":
        for name, module in model.named_modules():
            if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                parts = name.split(".")
                target_modules.add(parts[-1] if parts else name)
    
    # Default if no matches
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    return list(target_modules)


def get_model_chat_template(model_name: str) -> str:
    """Determine chat template from model name."""
    if "llama" in model_name.lower():
        return "llama"
    elif "qwen" in model_name.lower():
        return "qwen"
    else:
        return "qwen"  # Default


class CustomerServiceSFTTrainer:
    """
    SFT Trainer for customer service model.
    
    Supports both Qwen and Llama models with proper template handling.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Determine chat template
        self.chat_template = config.model.chat_template or get_model_chat_template(config.model.name)

    def setup_tokenizer(self):
        """Initialize tokenizer with proper settings."""
        print(f"Loading tokenizer: {self.config.model.name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            revision=self.config.model.revision,
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side="right",
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Chat template: {self.chat_template}")
        return self.tokenizer

    def setup_model(self):
        """Initialize model with appropriate quantization/LoRA settings."""
        print(f"Loading model: {self.config.model.name}")
        print(f"Revision: {self.config.model.revision}")
        
        # Setup tokenizer first
        if self.tokenizer is None:
            self.setup_tokenizer()
        
        # Quantization config
        from transformers import BitsAndBytesConfig
        quantization_config = None
        
        if self.config.quantization.enabled and self.config.quantization.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, 
                    self.config.quantization.bnb_4bit_compute_dtype,
                    torch.bfloat16
                ),
                bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
            )
        
        # Load model
        torch_dtype = torch.bfloat16 if self.config.hardware.bf16 else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            revision=self.config.model.revision,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        
        # Prepare for kbit training if quantized
        if self.config.quantization.enabled:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Get target modules and setup LoRA
        target_modules_str = self.config.lora.target_modules
        
        # Parse target modules
        if target_modules_str == "all-linear":
            target_modules = get_target_modules(self.model, self.chat_template)
        elif target_modules_str == "all":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in target_modules_str.split(",")]
        
        print(f"Target modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=target_modules,
            bias=self.config.lora.bias,
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer

    def setup_trainer(self, train_dataset, eval_dataset=None):
        """Setup SFTTrainer with proper collator."""
        # Create output directory
        Path(self.config.output.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            max_steps=-1,
            gradient_checkpointing=self.config.hardware.gradient_checkpointing,
            bf16=self.config.hardware.bf16,
            fp16=self.config.hardware.fp16,
            evaluation_strategy=self.config.evaluation.eval_strategy,
            eval_steps=self.config.evaluation.eval_steps if eval_dataset else None,
            save_strategy=self.config.evaluation.save_strategy,
            save_steps=self.config.evaluation.save_steps,
            save_total_limit=self.config.evaluation.save_total_limit,
            load_best_model_at_end=self.config.evaluation.load_best_model_at_end,
            metric_for_best_model=self.config.evaluation.metric_for_best_model,
            logging_steps=10,
            report_to="tensorboard",
            seed=self.config.hardware.seed,
            remove_unused_columns=False,
            dataloader_num_workers=self.config.hardware.dataloader_num_workers,
        )
        
        # Data collator for completion modeling
        # Use model-specific template
        if self.chat_template == "llama":
            response_template = "<|eot_id|>"  # Llama 3 special token
        else:
            response_template = "<|im_start|>assistant\n"
        
        data_collator = DataCollatorForCompletionLM(
            response_template=response_template,
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.training.max_seq_length,
            data_collator=data_collator,
        )
        
        return self.trainer

    def train(
        self, 
        train_data_path: str, 
        eval_data_path: Optional[str] = None
    ):
        """Run training."""
        print("=" * 60)
        print("SFT Training Configuration")
        print("=" * 60)
        print(f"Model: {self.config.model.name}")
        print(f"Chat Template: {self.chat_template}")
        print(f"Output: {self.config.output.output_dir}")
        print("=" * 60)
        
        # Setup model
        self.setup_model()
        
        # Load data
        print(f"\nLoading training data: {train_data_path}")
        train_dataset = load_conversation_dataset(
            train_data_path,
            self.tokenizer,
            self.chat_template,
            self.config.data.max_samples,
        )
        
        eval_dataset = None
        if eval_data_path and Path(eval_data_path).exists():
            print(f"Loading evaluation data: {eval_data_path}")
            eval_dataset = load_conversation_dataset(
                eval_data_path,
                self.tokenizer,
                self.chat_template,
                self.config.data.max_samples,
            )
        
        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Evaluation samples: {len(eval_dataset)}")
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Train
        print("\nStarting training...")
        self.trainer.train()
        
        # Save
        print("\nSaving model...")
        self.trainer.save_model()
        self.trainer.save_state()
        
        print("\nTraining complete!")
        return self.trainer


def train_sft(
    config_path: Optional[str] = None,
    model_key: Optional[str] = None,
    train_data: Optional[str] = None,
    eval_data: Optional[str] = None,
    **overrides
):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML config file
        model_key: Model key from MODEL_MATRIX (qwen3_8b, llama3_8b, etc.)
        train_data: Training data path
        eval_data: Evaluation data path
        **overrides: Config field overrides
    """
    # Load config
    if config_path:
        config = SFTConfig.from_yaml(config_path)
    elif model_key:
        config = create_config_for_model(model_key, **overrides)
    else:
        # Default Qwen config
        config = SFTConfig.from_dict({
            "model": {"name": "Qwen/Qwen3-8B", "chat_template": "qwen"},
            "lora": {"r": 16, "target_modules": "all-linear"},
        })
    
    # Override paths
    if train_data:
        config.data.train_path = train_data
    if eval_data:
        config.data.eval_path = eval_data
    
    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        return 1
    
    # Create trainer
    trainer = CustomerServiceSFTTrainer(config)
    
    # Run
    trainer.train(
        train_data_path=config.data.train_path,
        eval_data_path=config.data.eval_path,
    )
    
    return 0


# Import helper
from .sft_config import create_config_for_model


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--model", type=str, choices=list(MODEL_MATRIX.keys()), 
                       help="Model key (qwen3_8b, llama3_8b, etc.)")
    parser.add_argument("--train_data", type=str, help="Training data path")
    parser.add_argument("--eval_data", type=str, help="Evaluation data path")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Dry run without training")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run mode - validating config only")
        # Just load and validate
        if args.config:
            config = SFTConfig.from_yaml(args.config)
        elif args.model:
            config = create_config_for_model(args.model)
        else:
            config = SFTConfig()
        
        errors = config.validate()
        if errors:
            print("Errors found:")
            for e in errors:
                print(f"  - {e}")
            return 1
        
        print("Config valid!")
        print(json.dumps(config.to_dict(), indent=2, default=str))
        return 0
    
    return train_sft(
        config_path=args.config,
        model_key=args.model,
        train_data=args.train_data,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    exit(main())
