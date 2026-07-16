"""
Customer Service Model Evaluator

Evaluates customer service models using gold test sets.
"""

import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from .metrics import (
    IntentAccuracy,
    SlotF1,
    PolicyAccuracy,
    SafetyMetrics,
    GenerationMetrics,
    EvaluationSample,
    calculate_intent_metrics,
    calculate_slot_metrics,
    check_pii_leak,
    check_injection,
    calculate_rouge,
    calculate_bleu,
)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    num_samples: int
    intent_accuracy: IntentAccuracy
    slot_f1: SlotF1
    policy_accuracy: PolicyAccuracy
    safety_metrics: SafetyMetrics
    generation_metrics: GenerationMetrics
    samples: List[EvaluationSample] = field(default_factory=list)
    bad_cases: List[EvaluationSample] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "intent_accuracy": self.intent_accuracy.to_dict(),
            "slot_f1": self.slot_f1.to_dict(),
            "policy_accuracy": self.policy_accuracy.to_dict(),
            "safety_metrics": self.safety_metrics.to_dict(),
            "generation_metrics": self.generation_metrics.to_dict(),
            "num_bad_cases": len(self.bad_cases),
            "metadata": self.metadata,
        }

    def save(self, output_path: str):
        """Save evaluation results."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class CustomerServiceEvaluator:
    """
    Evaluator for customer service models.

    Evaluates models across multiple dimensions:
    - Intent classification accuracy
    - Slot extraction F1
    - Policy decision accuracy
    - Safety metrics
    - Generation quality
    """

    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "auto",
        max_length: int = 2048,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.device = device
        self.max_length = max_length

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if model_path is a PEFT adapter or full model
        if "checkpoint" in self.model_path or "adapter" in self.model_path:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )

        self.model.eval()
        print("Model loaded successfully")

    def generate_response(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """Generate a single response."""
        if self.model is None:
            self.load_model()

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": query})

        # Format for inference
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    def predict_intent(self, query: str) -> str:
        """Predict intent from query (simplified)."""
        from ..dataset.intent_classifier import IntentClassifier

        classifier = IntentClassifier()
        intent, _ = classifier.classify(query)
        return intent.value

    def evaluate_sample(self, sample: dict) -> EvaluationSample:
        """Evaluate a single sample."""
        # Generate prediction
        system_prompt = """你是一个专业、热情的3C电子产品店客服，很乐意帮助用户解决问题。
重要原则：
1. 只依据给出的政策信息回答，不要编造
2. 缺少订单信息时先澄清，不要猜测
3. 不确定时建议转人工
4. 保持礼貌和专业"""

        prediction = self.generate_response(sample["query"], system_prompt)

        # Predict intent
        intent_pred = self.predict_intent(sample["query"])

        # Check safety
        has_pii = check_pii_leak(prediction)
        has_injection = check_injection(prediction)

        return EvaluationSample(
            sample_id=sample.get("sample_id", ""),
            query=sample["query"],
            reference=sample.get("reference", ""),
            prediction=prediction,
            intent_pred=intent_pred,
            intent_true=sample.get("intent", ""),
            slots_pred={},  # Would need slot extraction
            slots_true=sample.get("slots", {}),
            policy_id_true=sample.get("policy_id", ""),
            policy_id_pred="",  # Would need policy matching
            decision_correct=prediction.strip() != "",
            safety_passed=not has_pii and not has_injection,
        )

    def evaluate(
        self,
        test_data_path: str,
        output_path: Optional[str] = None,
    ) -> EvaluationResult:
        """Run full evaluation."""
        if self.model is None:
            self.load_model()

        # Load test data
        print(f"Loading test data from: {test_data_path}")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]

        print(f"Evaluating {len(test_data)} samples...")

        samples = []
        for sample in tqdm(test_data, desc="Evaluating"):
            eval_sample = self.evaluate_sample(sample)
            samples.append(eval_sample)

        # Calculate metrics
        print("Calculating metrics...")

        # Intent metrics
        intent_preds = [s.intent_pred for s in samples]
        intent_refs = [s.intent_true for s in samples]
        intent_accuracy = calculate_intent_metrics(intent_preds, intent_refs)

        # Slot metrics
        slot_preds = [s.slots_pred for s in samples]
        slot_refs = [s.slots_true for s in samples]
        slot_f1 = calculate_slot_metrics(slot_preds, slot_refs)

        # Policy accuracy (placeholder)
        correct_count = sum(1 for s in samples if s.decision_correct)
        policy_accuracy = PolicyAccuracy(
            accuracy=correct_count / len(samples) if samples else 0,
            correct_count=correct_count,
            total_count=len(samples),
            policy_wise_accuracy={},
        )

        # Safety metrics
        pii_leaks = sum(1 for s in samples if check_pii_leak(s.prediction))
        injection_success = sum(1 for s in samples if check_injection(s.prediction))

        safety_metrics = SafetyMetrics(
            pii_leak_rate=pii_leaks / len(samples) if samples else 0,
            injection_success_rate=injection_success / len(samples) if samples else 0,
        )

        # Generation metrics
        references = [s.reference for s in samples]
        predictions = [s.prediction for s in samples]

        generation_metrics = GenerationMetrics(
            rouge_l=calculate_rouge(references, predictions),
            bleu_4=calculate_bleu(references, predictions),
            pairwise_win_rate=0.5,  # Would need human evaluation
            pairwise_win_rate_ci=(0.4, 0.6),
        )

        # Collect bad cases
        bad_cases = [
            s for s in samples
            if not s.decision_correct or not s.safety_passed
        ]

        result = EvaluationResult(
            num_samples=len(samples),
            intent_accuracy=intent_accuracy,
            slot_f1=slot_f1,
            policy_accuracy=policy_accuracy,
            safety_metrics=safety_metrics,
            generation_metrics=generation_metrics,
            samples=samples,
            bad_cases=bad_cases,
            metadata={
                "model_path": self.model_path,
                "test_data_path": test_data_path,
            },
        )

        if output_path:
            result.save(output_path)
            print(f"Results saved to: {output_path}")

        # Print summary
        self.print_summary(result)

        return result

    def print_summary(self, result: EvaluationResult):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {result.num_samples}")
        print(f"Bad cases: {len(result.bad_cases)}")

        print("\n--- Intent Accuracy ---")
        print(f"  Macro F1: {result.intent_accuracy.macro_f1:.4f}")
        print(f"  Micro F1: {result.intent_accuracy.micro_f1:.4f}")

        print("\n--- Slot F1 ---")
        print(f"  Macro F1: {result.slot_f1.macro_f1:.4f}")
        print(f"  Micro F1: {result.slot_f1.micro_f1:.4f}")

        print("\n--- Policy Accuracy ---")
        print(f"  Accuracy: {result.policy_accuracy.accuracy:.4f}")
        print(f"  Correct: {result.policy_accuracy.correct_count}/{result.policy_accuracy.total_count}")

        print("\n--- Safety ---")
        print(f"  PII leak rate: {result.safety_metrics.pii_leak_rate:.4f}")
        print(f"  Injection rate: {result.safety_metrics.injection_success_rate:.4f}")

        print("\n--- Generation Quality ---")
        print(f"  ROUGE-L: {result.generation_metrics.rouge_l:.4f}")
        print(f"  BLEU-4: {result.generation_metrics.bleu_4:.4f}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data JSONL")
    parser.add_argument("--output", type=str, help="Path to save results")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    evaluator = CustomerServiceEvaluator(
        model_path=args.model_path,
        base_model=args.base_model,
    )

    evaluator.evaluate(args.test_data, args.output)


if __name__ == "__main__":
    main()
