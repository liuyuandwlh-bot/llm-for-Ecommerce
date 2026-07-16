"""
Customer Service Model Evaluator

Evaluates customer service models using gold test sets.
Supports both real model evaluation and fixture-based testing.
"""

import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
            "intent_accuracy": {
                "accuracy": self.intent_accuracy.accuracy,
                "macro_f1": self.intent_accuracy.macro_f1,
                "micro_f1": self.intent_accuracy.micro_f1,
            },
            "slot_f1": {
                "macro_f1": self.slot_f1.macro_f1,
                "micro_f1": self.slot_f1.micro_f1,
            },
            "policy_accuracy": {
                "accuracy": self.policy_accuracy.accuracy,
                "correct": self.policy_accuracy.correct_count,
                "total": self.policy_accuracy.total_count,
            },
            "safety_metrics": {
                "pii_leak_rate": self.safety_metrics.pii_leak_rate,
                "injection_rate": self.safety_metrics.injection_success_rate,
            },
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
        model_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen3-8B",
        device: str = "auto",
        max_length: int = 2048,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.max_length = max_length
        
        # Parse device string
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self._is_mock = True

    def load_model(self):
        """Load model and tokenizer."""
        if self.model_path is None:
            print("No model path provided - using mock mode")
            self._is_mock = True
            return
        
        print(f"Loading model from: {self.model_path}")
        
        self._is_mock = False
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if self.model_path and ("checkpoint" in str(self.model_path) or "adapter" in str(self.model_path)):
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
        if self._is_mock:
            return f"[MOCK] 已收到您的问题：{query}"
        
        if self.model is None:
            self.load_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
        """Predict intent from query (placeholder - should use trained classifier)."""
        # Simple keyword-based prediction for demo
        if any(kw in query for kw in ["退货", "退款", "退"]):
            return "return_query"
        elif any(kw in query for kw in ["换货", "换"]):
            return "exchange_query"
        elif any(kw in query for kw in ["物流", "快递", "发货", "签收"]):
            return "logistics_query"
        elif any(kw in query for kw in ["优惠", "降价", "差价", "优惠券"]):
            return "coupon_or_price_protection"
        elif any(kw in query for kw in ["投诉", "经理"]):
            return "complaint"
        elif any(kw in query for kw in ["兼容", "支持"]):
            return "specification_query"
        else:
            return "unknown"

    def evaluate_sample(self, sample: dict) -> EvaluationSample:
        """Evaluate a single sample."""
        query = sample.get("query", sample.get("messages", [{}])[0].get("content", ""))
        
        # Generate prediction
        system_prompt = """你是一个专业、热情的3C电子产品店客服，很乐意帮助用户解决问题。
重要原则：
1. 只依据给出的政策信息回答，不要编造
2. 缺少订单信息时先澄清，不要猜测
3. 不确定时建议转人工
4. 保持礼貌和专业"""

        prediction = self.generate_response(query, system_prompt)

        # Predict intent
        intent_pred = self.predict_intent(query)
        intent_true = sample.get("intent", "")

        # Check safety
        has_pii = check_pii_leak(prediction)
        has_injection = check_injection(prediction)

        return EvaluationSample(
            sample_id=sample.get("sample_id", ""),
            query=query,
            reference=sample.get("reference", ""),
            prediction=prediction,
            intent_pred=intent_pred,
            intent_true=intent_true,
            slots_pred={},
            slots_true=sample.get("slots", {}),
            policy_id_true=sample.get("policy_ids", [""])[0] if sample.get("policy_ids") else "",
            policy_id_pred="",
            decision_correct=self._check_decision(prediction, sample.get("decision", "")),
            safety_passed=not has_pii and not has_injection,
        )

    def _check_decision(self, prediction: str, expected_decision: str) -> bool:
        """Check if prediction matches expected decision."""
        # Simple heuristic - in production use proper NER/classification
        if expected_decision == "full_refund":
            return "退款" in prediction or "退货" in prediction
        elif expected_decision == "reject":
            return "无法" in prediction or "抱歉" in prediction
        elif expected_decision == "escalate":
            return "转接" in prediction or "人工" in prediction
        return len(prediction) > 0

    def evaluate(
        self,
        test_data_path: str,
        output_path: Optional[str] = None,
    ) -> EvaluationResult:
        """Run full evaluation."""
        # Load model if needed
        if self.model is None:
            self.load_model()

        # Load test data
        print(f"Loading test data from: {test_data_path}")
        
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))

        print(f"Evaluating {len(test_data)} samples...")

        samples = []
        for sample in test_data:
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

        # Policy accuracy
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
        generation_metrics = GenerationMetrics(
            rouge_l=0.0,
            bleu_4=0.0,
            pairwise_win_rate=0.0,
            pairwise_win_rate_ci=(0.0, 0.0),
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
                "model_path": self.model_path or "mock",
                "test_data_path": test_data_path,
                "is_mock": self._is_mock,
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
        print(f"Mode: {'MOCK' if result.metadata.get('is_mock') else 'REAL MODEL'}")

        print("\n--- Intent Accuracy ---")
        print(f"  Accuracy: {result.intent_accuracy.accuracy:.4f}")
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

        print("\nNote: ROUGE/BLEU/pairwise metrics require reference responses and human evaluation.")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate customer service model")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, help="Path to test data JSONL")
    parser.add_argument("--output", type=str, help="Path to save results")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    
    args = parser.parse_args()
    
    evaluator = CustomerServiceEvaluator(
        model_path=args.model_path,
        base_model=args.base_model,
    )
    
    # Use fixture if no test data provided
    test_data = args.test_data or "data/fixtures/ecommerce/canonical_cases.jsonl"
    
    evaluator.evaluate(test_data, args.output)


if __name__ == "__main__":
    main()
