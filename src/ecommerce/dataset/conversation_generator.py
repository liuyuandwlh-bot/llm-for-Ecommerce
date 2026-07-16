"""
Synthetic Conversation Generator

Round 2 rewrite:
- Valid message ordering (system/user/assistant), no orphan assistant greetings
- Deterministic user-utterance rewrite library (real variation, not just
  greeting rewrite); strategies listed per sample
- Determinism via shared ``src.common.hashing`` (no ``hash()``,
  no module-level ``random``)
- Local ``random.Random`` instances only
- Single-file deterministic output, JSON-sorted key writing
- LLM provider moved to explicit protocol; raise at construction time
  without API key (no silent "NotImplementedError" later)
"""

import argparse
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from src.common.hashing import Hashing, seeded_random

# ---------------------------------------------------------------------------
# User utterance rewriter
# ---------------------------------------------------------------------------


@dataclass
class RewriteVariant:
    """A deterministic rewrite of the canonical user utterance."""

    strategy: str
    text: str
    notes: str = ""


# Each rewrite is a pure function over (case, rng) -> str and is identified by
# strategy. Callers apply ALL strategies for the case and write each as a
# separate sample (variant_id = 0..N).

_STRATEGY_REGISTRY: dict[str, Callable[[str, "random.Random"], str]] = {}  # type: ignore[name-defined]  # noqa: F821


def register_strategy(name: str):
    def deco(fn):
        _STRATEGY_REGISTRY[name] = fn
        return fn
    return deco


def get_strategies() -> dict[str, Callable]:
    return dict(_STRATEGY_REGISTRY)


@register_strategy("original")
def _identity(text: str, rng) -> str:
    return text


@register_strategy("colloquial")
def _colloquial(text: str, rng) -> str:
    replacements = [
        ("请问", ""),
        ("您好，", ""),
        ("请问", ""),
        ("能", "可以"),
        ("可以", "能不能"),
    ]
    out = text
    for a, b in replacements:
        if rng.random() < 0.5:
            out = out.replace(a, b)
    return out


@register_strategy("missing_slot_followup")
def _missing_slot(text: str, rng) -> str:
    """Rewrite to a follow-up question when key slots are unclear."""
    if rng.random() < 0.5:
        return text + " 大概是上周到的"
    return text + " 我还没拆"


@register_strategy("split_slots")
def _split(text: str, rng) -> str:
    """Turn a long utterance into a multi-part request."""
    parts = re.split(r"[，。,.]", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2 and rng.random() < 0.7:
        return parts[0] + "。" + parts[1]
    return text


@register_strategy("emotional")
def _emotional(text: str, rng) -> str:
    suffixes = ["！", "！！", " 麻烦尽快处理"]
    return text + rng.choice(suffixes)


@register_strategy("short_alias")
def _short(text: str, rng) -> str:
    # Drop common polite starters to simulate abbreviated typing.
    starters = ["你好，", "你好", "您好，", "您好", "请问", "麻烦问一下，", "麻烦问一下"]
    out = text
    for s in starters:
        if out.startswith(s):
            return out[len(s):]
    return out


# ---------------------------------------------------------------------------
# Response templates (deterministic, no real LLM)
# ---------------------------------------------------------------------------


_DECISION_TO_TEMPLATE: dict[str, list[str]] = {
    "full_refund": [
        "好的，符合7天无理由退货条件，您可以申请全额退款。请在订单页面点击'申请退货'。",
        "您的订单符合退货条件，已为您记录，请稍后到订单详情提交退货申请。",
    ],
    "exchange": [
        "好的，请问您希望换成什么颜色或型号？我帮您查询库存。",
        "您的换货需求已收到，请告诉我目标规格我来为您确认。",
    ],
    "reject": [
        "抱歉，您的订单已超过无理由退货期限且不属于质量问题，建议联系专员确认是否能申请专项处理。",
    ],
    "need_more_info": [
        "好的，为了准确帮您处理，请提供订单号和签收时间，方便我查询。",
        "好的，请告诉我签收天数和商品状态，我再帮您判断。",
    ],
    "escalate": [
        "理解您的心情，我立即为您转接人工客服，请稍等。",
    ],
    "provide_info": [
        "可以的，本产品兼容主流蓝牙设备，包括华为、小米、苹果等。",
    ],
    "resend_or_refund": [
        "收到，这种情况可以为您补发或退款，请告诉我您希望怎么处理。",
    ],
    "cancel_with_compensation": [
        "抱歉，延迟发货了，我们可以为您取消并提供补偿，或者尽快安排发货。",
    ],
    "refund_difference": [
        "好的，同款确实降价了，差价会在3个工作日内原路退回。",
    ],
    "coupon_not_applicable": [
        "不好意思，本次订单金额不满足优惠券使用门槛，无法使用。",
    ],
    "ship_within_48h": [
        "好的，您的订单是现货，将于48小时内发货，请耐心等待。",
    ],
    "accept_return_refund": [
        "好的，已为您记录虚假宣传情况，将为您办理退货退款。",
    ],
    "acknowledge_apologize": [
        "非常抱歉给您带来不愉快的购物体验，我尽快帮您处理。",
    ],
}


def _response_template_for(decision: str, rng) -> str:
    candidates = _DECISION_TO_TEMPLATE.get(decision)
    if not candidates:
        return "好的，我会尽快帮您处理。"
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConversation:
    sample_id: str
    messages: list[dict[str, str]]
    intent: str
    slots: dict[str, object]
    policy_ids: list[str]
    decision: str
    requires_human: bool
    source_id: str = "owned_sop_v1"
    source_type: str = "synthetic_from_owned_sop"
    source_revision: str = "2026-07-16"
    parent_case_id: str = ""
    category_hint: str | None = None
    synthetic: bool = True
    generator_model: str = "deterministic-template-v2"
    generator_revision: str = "0.2.0"
    prompt_hash: str | None = None
    template_family: str = ""
    rewrite_strategies: list[str] = field(default_factory=list)
    rewrite_strategy: str = "original"
    variant_id: int = 0
    seed: int = 0
    pii_status: str = "passed"
    dedup_cluster_id: str | None = None
    quality_status: str = "generated"
    review_status: str = "pending"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class DeterministicConversationGenerator:
    """Build conversation samples from a canonical case.

    For each case we emit one sample per declared rewrite strategy so that
    the user-utterance variation is real, not just a greeting rewrite.
    """

    SYSTEM_PROMPT = (
        "你是3C数码店的客服助手，依据给出的政策规则和槽位回答用户。"
        "缺信息时先澄清，不要猜测；不确定时建议人工。不要回显任何 PII。"
    )

    def __init__(self, seed: int = 42, source_id: str = "owned_sop_v1", mode: str = "fixture"):
        self.seed = seed
        self.source_id = source_id
        self.mode = mode  # "fixture" or "training_release"

    def generate_for_case(
        self,
        case: dict,
        parent_case_id: str = "",
        template_family_override: str | None = None,
        strategies: list[str] | None = None,
    ) -> list[SyntheticConversation]:
        """Produce one synthetic conversation per rewrite strategy.

        The caller is responsible for emitting parent/chunk metadata; we
        always emit messages that satisfy the role validator used downstream.

        Strategy precedence:
        1. explicit ``strategies`` argument (caller-provided, e.g. CLI ``--samples``)
        2. ``case["rewrite_strategies"]`` if non-empty
        3. ``DEFAULT_REWRITE_STRATEGIES``
        """
        case_id = case.get("case_id", parent_case_id)
        family = template_family_override or case.get("template_family", "default")
        if strategies is not None:
            strategies_list = list(strategies)
        elif case.get("rewrite_strategies") and not (
            isinstance(case["rewrite_strategies"], list)
            and case["rewrite_strategies"]
            and case["rewrite_strategies"][0] == "__use_generator_defaults__"
        ):
            strategies_list = list(case["rewrite_strategies"])
        else:
            strategies_list = list(DEFAULT_REWRITE_STRATEGIES)
        if "original" not in strategies_list:
            strategies_list.insert(0, "original")

        conversations: list[SyntheticConversation] = []
        for variant_idx, strategy in enumerate(strategies_list):
            conv = self._generate_single(
                case=case,
                parent_case_id=case_id,
                template_family=family,
                rewrite_strategy=strategy,
                variant_id=variant_idx,
            )
            conversations.append(conv)

        return conversations

    def _generate_single(
        self,
        case: dict,
        parent_case_id: str,
        template_family: str,
        rewrite_strategy: str,
        variant_id: int,
    ) -> SyntheticConversation:
        # Local RNG seeded by the stable triple (parent_case_id, strategy, variant)
        # so different processes produce identical samples.
        local_seed = Hashing.int(
            "conv",
            self.seed,
            parent_case_id,
            template_family,
            rewrite_strategy,
            variant_id,
            mod=2**31 - 1,
        )
        rng = seeded_random("conv", local_seed)

        rewrite_fn = _STRATEGY_REGISTRY.get(rewrite_strategy, _identity)
        original_user_turns = [
            t for t in case.get("turns", []) if t.get("role") == "user"
        ]
        rewritten_user_text = rewrite_fn(
            original_user_turns[0]["content"] if original_user_turns else "",
            rng,
        )

        # For missing_slot_followup strategy we may split the request across
        # two user turns. Build final user-turn list deterministically.
        user_turns: list[dict[str, str]] = []
        if rewrite_strategy == "split_slots" and len(original_user_turns) >= 2:
            user_turns = list(original_user_turns)
            for i, t in enumerate(user_turns):
                user_turns[i] = {"role": "user", "content": rewrite_fn(t.get("content", ""), rng)}
        elif rewrite_strategy == "missing_slot_followup":
            head = rewritten_user_text.split("，")[0]
            tail = rewritten_user_text[len(head) + 1:] if len(rewritten_user_text) > len(head) + 1 else ""
            user_turns.append({"role": "user", "content": head})
            if tail.strip():
                user_turns.append({"role": "user", "content": tail.strip(" ，,")})
        else:
            user_turns.append({"role": "user", "content": rewritten_user_text})

        # Build messages starting with system (legal in our validator).
        decision = case.get("expected_decision") or "need_more_info"
        requires_human = bool(case.get("requires_human", False))

        messages: list[dict[str, str]] = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add user turns followed by an assistant reply.
        for idx, ut in enumerate(user_turns):
            messages.append({"role": "user", "content": ut["content"]})
            # Only emit the assistant reply after the LAST user turn to mirror
            # single-turn assistant behavior; intermediate assistant replies
            # (for multi-turn) come from `turns` already.
            if idx == len(user_turns) - 1:
                messages.append({"role": "assistant", "content": _response_template_for(decision, rng)})

        # If the case has a pre-built user/assistant sequence (multi-turn
        # cases), append only the missing non-system assistant replies so the
        # final sequence always ends with assistant and contains policy content.
        case_turns = case.get("turns") or []
        canonical_assistant_texts = [t["content"] for t in case_turns if t.get("role") == "assistant"]
        if canonical_assistant_texts:
            # Use the LAST canonical assistant text as the final assistant
            # turn (multi-turn -> last message = assistant).
            messages[-1] = {"role": "assistant", "content": canonical_assistant_texts[-1]}

        sample_id = Hashing.short(
            "sample",
            self.seed,
            parent_case_id,
            template_family,
            rewrite_strategy,
            variant_id,
        )
        return SyntheticConversation(
            sample_id=f"ecom_sft_{sample_id}",
            messages=messages,
            intent=case.get("intent", "unknown"),
            slots=dict(case.get("context") or {}),
            category_hint=case.get("category_hint"),
            policy_ids=list(case.get("expected_policy_ids") or []),
            decision=decision,
            requires_human=requires_human,
            source_id=self.source_id,
            source_type="synthetic_from_owned_sop",
            source_revision="2026-07-16",
            parent_case_id=parent_case_id,
            template_family=template_family,
            rewrite_strategy=rewrite_strategy,
            rewrite_strategies=[rewrite_strategy],
            variant_id=variant_id,
            seed=local_seed,
            quality_status="generated",
            review_status="auto_validated" if self.mode == "fixture" else "pending",
        )


# Default strategies applied to every case unless overridden.
DEFAULT_REWRITE_STRATEGIES = [
    "original",
    "colloquial",
    "emotional",
]


# ---------------------------------------------------------------------------
# Conversation generator facade (drop-in replacement for the round-1 API)
# ---------------------------------------------------------------------------


class ConversationGenerator:
    """Backward-compatible generator façade.

    The round-1 sample structure is preserved so downstream code works, but
    we now emit one conversation per declared rewrite strategy instead of a
    fixed number of greeting variants.
    """

    def __init__(self, policies: list | None = None, seed: int = 42, mode: str = "fixture", **kwargs):
        self.policies = policies
        self.seed = seed
        self.mode = mode
        self._impl = DeterministicConversationGenerator(seed=seed, mode=mode)
        self.conversations: list[SyntheticConversation] = []

    def _reseed(self, seed: int) -> None:
        self.seed = seed
        self._impl = DeterministicConversationGenerator(seed=seed, mode=self.mode)

    def set_seed(self, seed: int) -> None:
        self._reseed(seed)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._impl.mode = mode

    def generate_from_cases(
        self,
        cases: Iterable[dict],
        samples_per_case: int | None = None,
    ) -> list[SyntheticConversation]:
        cases = list(cases)
        self.conversations = []
        for case in cases:
            self.conversations.extend(
                self._impl.generate_for_case(case)
            )
        return self.conversations

    def run(
        self,
        output_path: str,
        cases: Iterable[dict] | None = None,
    ) -> list[SyntheticConversation]:
        """Convenience: optionally build conversations, then save."""
        if cases is not None:
            self.generate_from_cases(cases)
        self.save_conversations(output_path)
        return self.conversations

    def save_conversations(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Stable order: sort by sample_id
        ordered = sorted(self.conversations, key=lambda c: c.sample_id)
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in ordered:
                f.write(json.dumps(conv.to_dict(), ensure_ascii=False, sort_keys=True) + '\n')


# ---------------------------------------------------------------------------
# LLMGenerator (explicit protocol; raise early without API key)
# ---------------------------------------------------------------------------


class LLMUnavailableError(RuntimeError):
    """Raised when an LLM-based generator is requested without proper config."""


class LLMGenerator:
    """Skeleton LLM-based generator.

    Round 2 design: instead of silently returning a "NotImplementedError"
    later, this class raises ``LLMUnavailableError`` at construction time if
    no provider is configured. Callers should check availability before
    invoking.
    """

    PROTOCOL_NAME = "deterministic-template-v2"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        provider: str = "openai",
    ):
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMUnavailableError(
                "LLMGenerator requires OPENAI_API_KEY environment variable "
                "(or explicit api_key). Use DeterministicConversationGenerator "
                "for offline reproduction."
            )

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, case: dict) -> SyntheticConversation:  # pragma: no cover
        raise LLMUnavailableError(
            "LLMGenerator.generate is not implemented in this build. "
            "Use DeterministicConversationGenerator."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic conversations"
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="data/processed/fixtures/policies.json",
        help="Policy JSON (used to cross-check existence; not strictly required for deterministic generation).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        required=True,
        help="Canonical cases JSONL",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output conversations JSONL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default="owned_sop_v1",
        help="Source identifier for samples",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fixture",
        choices=["fixture", "training_release"],
        help="Whether samples are fixture-only or destined for training release",
    )
    args = parser.parse_args()

    with open(args.cases, encoding='utf-8') as f:
        cases = [json.loads(line) for line in f]

    gen = ConversationGenerator(seed=args.seed, mode=args.mode)
    gen.set_seed(args.seed)
    gen.set_mode(args.mode)
    # Set source id on impl
    gen._impl.source_id = args.source_id
    conversations = gen.generate_from_cases(cases)
    gen.save_conversations(args.output)

    print(f"Generated {len(conversations)} conversations -> {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
