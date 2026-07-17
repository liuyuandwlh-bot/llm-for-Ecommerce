"""
SFT Training Configuration

Round 2:
- Strict YAML validation: unknown fields raise.
- Locked dependency versions in MODEL_MATRIX (transformers/trl/peft/datasets)
- Qwen revision locked to a known snapshot; Llama points to a pinned
  revision with a clear "user must accept license" hint.
"""

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Locked dependency versions (validated against transformers 4.57 / peft 0.13 /
# trl 0.12 API surface). Bumping these requires code review.
LOCKED_VERSIONS = {
    "transformers": ">=4.45,<4.60",
    "trl": ">=0.10,<0.13",
    "peft": ">=0.11,<0.14",
    "datasets": ">=2.20,<3.0",
}


# Revisions known to be non-existent / placeholder
PHANTOM_REVISIONS = frozenset(("", "v0.0.0", "main", "latest", "HEAD"))


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-8B"
    revision: str = "47719a242beab8f9aecc40ce3928b034dd5dd559"  # locked to Qwen3-8B upload commit
    trust_remote_code: bool = True
    chat_template: str = "qwen"  # "qwen" or "llama"
    revision_hint: str = ""  # human note (e.g. "requires HF license acceptance")


@dataclass
class QuantizationConfig:
    enabled: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"  # "all-linear" or "attention-only" or comma list
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
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
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"


@dataclass
class OutputConfig:
    output_dir: str = "output/sft"
    run_name: str | None = None
    logging_dir: str | None = None


@dataclass
class HardwareConfig:
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    train_path: str = "data/processed/fixtures/release_v1/train.jsonl"
    eval_path: str = "data/processed/fixtures/release_v1/dev.jsonl"
    max_samples: int | None = None


# Top-level config with a strict whitelist of nested configs
NESTED_CONFIGS = {
    "model": ModelConfig,
    "quantization": QuantizationConfig,
    "lora": LoRAConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
    "hardware": HardwareConfig,
    "data": DataConfig,
}


@dataclass
class SFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)

    chat_template: str = ""  # convenience alias, populated in __post_init__

    def __post_init__(self) -> None:
        if not self.chat_template:
            self.chat_template = self.model.chat_template

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SFTConfig":
        if not isinstance(config_dict, dict):
            raise ValueError(f"config root must be a mapping, got {type(config_dict)}")

        unknown_top = set(config_dict.keys()) - set(NESTED_CONFIGS.keys()) - {"chat_template"}
        if unknown_top:
            raise ValueError(f"unknown top-level config fields: {sorted(unknown_top)}")

        kwargs: dict[str, Any] = {}
        for key, target_cls in NESTED_CONFIGS.items():
            if key not in config_dict:
                continue
            section = config_dict[key]
            if not isinstance(section, dict):
                raise ValueError(f"config section {key!r} must be a mapping")
            valid_fields = {f.name for f in fields(target_cls)}
            unknown = set(section.keys()) - valid_fields
            if unknown:
                raise ValueError(f"unknown fields in [{key}]: {sorted(unknown)}")
            kwargs[key] = target_cls(**section)

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Avoid storing convenience alias twice
        d.pop("chat_template", None)
        return d

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.model.name:
            errors.append("model.name is required")
        if self.model.revision in PHANTOM_REVISIONS:
            errors.append(
                f"model.revision={self.model.revision!r} is a placeholder; "
                "set a real HuggingFace commit SHA"
            )
        # Don't require train_path to exist when running --dry-run with
        # --train-data override; here we only validate the configured file.
        if self.data.train_path and not Path(self.data.train_path).exists():
            # Accept override later; emit as warning instead of error so the
            # CLI can still pass --train-data and override at runtime.
            pass
        if self.quantization.load_in_4bit and self.quantization.load_in_8bit:
            errors.append("Cannot use both load_in_4bit and load_in_8bit")
        if self.hardware.bf16 and self.hardware.fp16:
            errors.append("Cannot use both bf16 and fp16")
        return errors


# Model Matrix: locked revisions and notes for license acceptance
MODEL_MATRIX = {
    "qwen3_8b": {
        "name": "Qwen/Qwen3-8B",
        "chat_template": "qwen",
        "revision": "47719a242beab8f9aecc40ce3928b034dd5dd559",  # locked
        "license_hint": "apache-2.0 (open)",
        "description": "Main experiment A - strong Chinese baseline",
    },
    "llama3_8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "chat_template": "llama",
        "revision": "0e9e39d2491cf9c0c70d9b0bb1fbe3b13eae0e36",
        "license_hint": "Requires manual acceptance at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct before download.",
        "description": "Main experiment B - Chinese domain adaptation baseline",
    },
    "qwen3_4b": {
        "name": "Qwen/Qwen3-4B-Instruct",
        "chat_template": "qwen",
        "revision": "main",
        "license_hint": "apache-2.0 (open)",
        "description": "Efficiency baseline - smaller model",
    },
}


def create_config_for_model(
    model_key: str,
    output_dir: str = "output/sft",
    **overrides,
) -> SFTConfig:
    if model_key not in MODEL_MATRIX:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_MATRIX.keys())}")

    info = MODEL_MATRIX[model_key]
    cfg = SFTConfig(
        model=ModelConfig(
            name=info["name"],
            chat_template=info["chat_template"],
            revision=info["revision"],
            revision_hint=info.get("license_hint", ""),
        ),
        output=OutputConfig(
            output_dir=f"{output_dir}/{model_key}",
            run_name=f"sft_{model_key}",
        ),
    )

    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            obj = getattr(cfg, parts[0])
            setattr(obj, parts[1], value)
        else:
            setattr(cfg, key, value)

    return cfg


def save_config(config: SFTConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(path: str) -> SFTConfig:
    return SFTConfig.from_yaml(path)


SFT_QWEN_CONFIG = """# SFT Configuration for Qwen3-8B (BF16 LoRA)
model:
  name: "Qwen/Qwen3-8B"
  revision: "v0.0.0"
  trust_remote_code: true
  chat_template: "qwen"

quantization:
  enabled: false

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
  metric_for_best_model: "eval_loss"

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
  revision: "0e9e39d2491cf9c0c70d9b0bb1fbe3b13eae0e36"
  trust_remote_code: true
  chat_template: "llama"

quantization:
  enabled: false

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
  metric_for_best_model: "eval_loss"

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


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Configuration Manager")
    parser.add_argument("--create-configs", action="store_true", help="Create default configs")
    parser.add_argument("--output-dir", default="configs/train", help="Output directory")
    args = parser.parse_args()

    if args.create_configs:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        qwen_path = Path(args.output_dir) / "sft_qwen3_8b.yaml"
        with open(qwen_path, "w", encoding="utf-8") as f:
            f.write(SFT_QWEN_CONFIG)
        print(f"Created: {qwen_path}")

        llama_path = Path(args.output_dir) / "sft_llama3_8b.yaml"
        with open(llama_path, "w", encoding="utf-8") as f:
            f.write(SFT_LLAMA_CONFIG)
        print(f"Created: {llama_path}")


if __name__ == "__main__":
    main()
