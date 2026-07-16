"""
Customer-service evaluator - Round 2 rewrite.

Goals:
- The CLI works without torch/transformers installed (lazy imports).
- Mock and oracle backends are clearly labelled in the output and are NOT
  promoted as model performance.
- All metrics come from `src.ecommerce.eval.metrics.summarize_evaluation`.
- Both SFT-style messages and canonical `turns` are supported as input.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import (
    Prediction,
    parse_prediction,
    summarize_evaluation,
)

_BACKEND_TYPES = ("fixture_oracle", "mock", "real_model")


@dataclass
class EvaluatorConfig:
    backend: str = "fixture_oracle"
    base_model: str = ""
    seed: int = 42
    generation_config: dict[str, Any] = field(default_factory=dict)


def _extract_first_user_text(sample: dict[str, Any]) -> str:
    """Pull the user-side text from either `messages` or `turns`."""
    if sample.get("messages"):
        for m in sample["messages"]:
            if m.get("role") == "user":
                return m.get("content") or ""
    if sample.get("turns"):
        for m in sample["turns"]:
            if m.get("role") == "user":
                return m.get("content") or ""
    return sample.get("query") or sample.get("user_text") or ""


def _to_reference(sample: dict[str, Any]) -> dict[str, Any]:
    """Build the reference dict used by ``summarize_evaluation``."""
    return {
        "intent": sample.get("intent", "unknown"),
        "slots": sample.get("slots") or sample.get("context") or {},
        "policy_ids": list(sample.get("policy_ids") or sample.get("expected_policy_ids") or []),
        "decision": sample.get("decision") or sample.get("expected_decision") or "need_more_info",
        "expected_missing_slots": list(sample.get("expected_missing_slots") or sample.get("missing_slots") or []),
        "requires_human": bool(sample.get("requires_human", False)),
        "tool_expectation_dict": sample.get("tool_expectation_dict"),
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _prediction_to_dict(p: Prediction) -> dict[str, Any]:
    return {
        "intent": p.intent,
        "slots": dict(p.slots or {}),
        "policy_ids": list(p.policy_ids or []),
        "decision": p.decision,
        "missing_slots": list(p.missing_slots or []),
        "requires_human": bool(p.requires_human),
        "tool_call": dict(p.tool_call) if isinstance(p.tool_call, dict) else None,
        "response": str(p.response or ""),
    }


def _fixture_oracle_predict(sample: dict[str, Any]) -> tuple[bool, Prediction | None, str | None]:
    """Use the canonical labels as the prediction. Returns (ok, pred, error)."""
    try:
        pred = Prediction(
            intent=sample.get("intent"),
            slots=dict(sample.get("slots") or sample.get("context") or {}),
            policy_ids=list(sample.get("policy_ids") or sample.get("expected_policy_ids") or []),
            decision=sample.get("decision") or sample.get("expected_decision") or "need_more_info",
            missing_slots=list(sample.get("expected_missing_slots") or []),
            requires_human=bool(sample.get("requires_human", False)),
            tool_call=None,
            response=sample.get("response", ""),
        )
        return True, pred, None
    except Exception as exc:  # pragma: no cover
        return False, None, str(exc)


def _mock_predict(sample: dict[str, Any]) -> tuple[bool, Prediction | None, str | None]:
    """Mock backend: deterministic placeholder predictions.

    Returns a partial prediction with intent guessed from keyword overlap and
    ``decision`` left blank so the metric reports a parse failure or a
    wrong decision - never a synthetic correct answer.
    """
    text = _extract_first_user_text(sample)
    intent = "unknown"
    if any(k in text for k in ("退", "退款")):
        intent = "return_query"
    elif any(k in text for k in ("换",)):
        intent = "exchange_query"
    elif any(k in text for k in ("物流", "发货", "快递")):
        intent = "logistics_query"
    elif any(k in text for k in ("价", "券", "优惠")):
        intent = "coupon_or_price_protection"
    elif any(k in text for k in ("兼容", "支持")):
        intent = "specification_query"
    elif any(k in text for k in ("投诉", "经理")):
        intent = "complaint"
    pred = Prediction(
        intent=intent,
        slots={},
        policy_ids=[],
        decision=None,
        missing_slots=[],
        requires_human=False,
        tool_call=None,
        response="[MOCK]",
    )
    return True, pred, None


def _real_model_predict(
    sample: dict[str, Any],
    model: Any = None,
    tokenizer: Any = None,
    generation_config: dict[str, Any] | None = None,
) -> tuple[bool, Prediction | None, str | None]:
    """Real-model backend. Heavy imports happen lazily."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        return False, None, f"real_model backend requires torch ({exc})"
    if model is None or tokenizer is None:
        return False, None, "real_model backend requires model and tokenizer"

    text = _extract_first_user_text(sample)
    messages = [
        {"role": "system", "content": "你是3C数码店客服助手，请根据用户问题预测结构化字段。"},
        {"role": "user", "content": text or ""},
    ]
    try:
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_cfg = {
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 0.0,
        }
        if generation_config:
            gen_cfg.update(generation_config)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_cfg)
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as exc:  # pragma: no cover
        return False, None, f"generation failed: {exc}"
    return parse_prediction(decoded)


# ---------------------------------------------------------------------------
# Loader / runner
# ---------------------------------------------------------------------------


def load_test_set(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_evaluation(
    test_path: str,
    backend: str = "fixture_oracle",
    output_dir: str | None = None,
    generation_config: dict[str, Any] | None = None,
    model_path: str | None = None,
    base_model: str = "Qwen/Qwen3-8B",
    seed: int = 42,
) -> dict[str, Any]:
    if backend not in _BACKEND_TYPES:
        raise ValueError(f"unknown backend: {backend}")

    samples = load_test_set(test_path)

    records: list[dict[str, Any]] = []
    for sample in samples:
        if backend == "fixture_oracle":
            ok, pred, err = _fixture_oracle_predict(sample)
        elif backend == "mock":
            ok, pred, err = _mock_predict(sample)
        else:
            ok, pred, err = _real_model_predict(
                sample,
                model=getattr(run_evaluation, "_model", None),
                tokenizer=getattr(run_evaluation, "_tokenizer", None),
                generation_config=generation_config,
            )
        records.append(
            {
                "sample_id": sample.get("sample_id") or sample.get("case_id") or f"sample-{len(records)}",
                "parsed": _prediction_to_dict(pred) if (ok and pred is not None) else None,
                "parse_error": err,
                "reference": _to_reference(sample),
            }
        )

    summary = summarize_evaluation(records)
    summary["backend_type"] = backend
    summary["is_model_result"] = backend == "real_model"
    summary["seed"] = seed
    summary["base_model"] = base_model if backend == "real_model" else ""
    summary["test_path"] = test_path
    summary["n_samples"] = len(samples)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        badcases = [
            r for r in records
            if (not r.get("parsed")) or (
                r["parsed"]
                and (
                    r["parsed"]["decision"] != r["reference"]["decision"]
                    or set(r["parsed"]["policy_ids"]) != set(r["reference"]["policy_ids"])
                    or r["parsed"]["intent"] != r["reference"]["intent"]
                )
            )
        ]
        with open(out / "badcases.jsonl", "w", encoding="utf-8") as f:
            for r in badcases:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        with open(out / "parse_failures.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                if not r.get("parsed"):
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Customer service evaluator")
    parser.add_argument("--backend", choices=_BACKEND_TYPES, default="fixture_oracle")
    parser.add_argument("--test-data", required=True, help="Test JSONL path")
    parser.add_argument("--output-dir", help="Directory for summary/predictions/badcases")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-path", default=None, help="Checkpoint for real_model backend")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        summary = run_evaluation(
            test_path=args.test_data,
            backend=args.backend,
            output_dir=args.output_dir,
            model_path=args.model_path,
            base_model=args.base_model,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"evaluation failed: {exc}")
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    exit(main())
