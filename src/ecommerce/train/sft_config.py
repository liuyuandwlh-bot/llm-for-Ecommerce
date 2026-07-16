"""
SFT Training Configuration and Launcher

Supports both Qwen and Llama models with proper template handling.
"""

import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen3-8B"
    revision: str = "main"
    trust_remote_code: bool = True
    chat_template: str = "qwen"  # "qwen" or "llama"


@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    enabled: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 1536
    packing: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "output/sft"
    run_name: Optional[str] = None
    logging_dir: Optional[str] = None


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = "data/processed/fixtures/release_v1/train.jsonl"
    eval_path: str = "data/processed/fixtures/release_v1/dev.jsonl"
    max_samples: Optional[int] = None


@dataclass
class SFTConfig:
    """Complete SFT configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SFTConfig":
        """Create config from dictionary."""
        # Handle nested config structure
        if "model" in config_dict:
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "quantization" in config_dict:
            config_dict["quantization"] = QuantizationConfig(**config_dict["quantization"])
        if "lora" in config_dict:
            config_dict["lora"] = LoRAConfig(**config_dict["lora"])
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])
        if "evaluation" in config_dict:
            config_dict["evaluation"] = EvaluationConfig(**config_dict["evaluation"])
        if "output" in config_dict:
            config_dict["output"] = OutputConfig(**config_dict["output"])
        if "hardware" in config_dict:
            config_dict["hardware"] = HardwareConfig(**config_dict["hardware"])
        if "data" in config_dict:
            config_dict["data"] = DataConfig(**config_dict["data"])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check model name is valid
        if not self.model.name:
            errors.append("model.name is required")
        
        # Check paths exist
        if not Path(self.data.train_path).exists():
            errors.append(f"Training data not found: {self.data.train_path}")
        
        # Check incompatible settings
        if self.quantization.load_in_4bit and self.quantization.load_in_8bit:
            errors.append("Cannot use both load_in_4bit and load_in_8bit")
        
        if self.hardware.bf16 and self.hardware.fp16:
            errors.append("Cannot use both bf16 and fp16")
        
        return errors


# Model Matrix
MODEL_MATRIX = {
    "qwen3_8b": {
        "name": "Qwen/Qwen3-8B",
        "chat_template": "qwen",
        "description": "Main experiment A - strong Chinese baseline",
    },
    "llama3_8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "chat_template": "llama",
        "description": "Main experiment B - Chinese domain adaptation baseline",
    },
    "qwen3_4b": {
        "name": "Qwen/Qwen3-4B-Instruct",
        "chat_template": "qwen",
        "description": "Efficiency baseline - smaller model",
    },
}


def create_config_for_model(
    model_key: str,
    output_dir: str = "output/sft",
    **overrides
) -> SFTConfig:
    """Create configuration for a specific model."""
    if model_key not in MODEL_MATRIX:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_MATRIX.keys())}")
    
    model_info = MODEL_MATRIX[model_key]
    
    config = SFTConfig(
        model=ModelConfig(
            name=model_info["name"],
            chat_template=model_info["chat_template"],
        ),
        output=OutputConfig(
            output_dir=f"{output_dir}/{model_key}",
            run_name=f"sft_{model_key}",
        ),
    )
    
    # Apply overrides
    for key, value in overrides.items():
        if "." in key:
            # Nested field, e.g., "training.learning_rate"
            parts = key.split(".")
            obj = getattr(config, parts[0])
            setattr(obj, parts[1], value)
        else:
            setattr(config, key, value)
    
    return config


def save_config(config: SFTConfig, path: str):
    """Save configuration to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(path: str) -> SFTConfig:
    """Load configuration from YAML file."""
    return SFTConfig.from_yaml(path)


# Training YAML configs
SFT_QWEN_CONFIG = """# SFT Configuration for Qwen3-8B (BF16 LoRA)

model:
  name: "Qwen/Qwen3-8B"
  revision: "main"
  trust_remote_code: true
  chat_template: "qwen"

quantization:
  enabled: false  # BF16 LoRA, not QLoRA

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: "all-linear"
  bias: "none"

training:
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 1536
  packing: false

evaluation:
  eval_strategy: "steps"
  eval_steps: 100
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 3
  load_best_model_at_end: true

output:
  output_dir: "output/sft/qwen3_8b"
  run_name: "sft_qwen3_8b"

hardware:
  gradient_checkpointing: true
  bf16: true
  fp16: false
  seed: 42

data:
  train_path: "data/processed/fixtures/release_v1/train.jsonl"
  eval_path: "data/processed/fixtures/release_v1/dev.jsonl"
  max_samples: null
"""

SFT_LLAMA_CONFIG = """# SFT Configuration for Llama-3.1-8B (BF16 LoRA)

model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  revision: "main"
  trust_remote_code: true
  chat_template: "llama"

quantization:
  enabled: false  # BF16 LoRA, not QLoRA

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: "all-linear"  # Will be adjusted for Llama architecture
  bias: "none"

training:
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 1536
  packing: false

evaluation:
  eval_strategy: "steps"
  eval_steps: 100
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 3
  load_best_model_at_end: true

output:
  output_dir: "output/sft/llama3_8b"
  run_name: "sft_llama3_8b"

hardware:
  gradient_checkpointing: true
  bf16: true
  fp16: false
  seed: 42

data:
  train_path: "data/processed/fixtures/release_v1/train.jsonl"
  eval_path: "data/processed/fixtures/release_v1/dev.jsonl"
  max_samples: null
"""


def main():
    """CLI entry point for config management."""
    parser = argparse.ArgumentParser(description="SFT Configuration Manager")
    parser.add_argument("--create-configs", action="store_true", help="Create default configs")
    parser.add_argument("--output-dir", default="configs/train", help="Output directory")
    
    args = parser.parse_args()
    
    if args.create_configs:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save Qwen config
        qwen_path = Path(args.output_dir) / "sft_qwen3_8b.yaml"
        with open(qwen_path, 'w') as f:
            f.write(SFT_QWEN_CONFIG)
        print(f"Created: {qwen_path}")
        
        # Save Llama config
        llama_path = Path(args.output_dir) / "sft_llama3_8b.yaml"
        with open(llama_path, 'w') as f:
            f.write(SFT_LLAMA_CONFIG)
        print(f"Created: {llama_path}")


if __name__ == "__main__":
    main()
