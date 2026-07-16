"""
Synthetic Conversation Generator

Generates multi-turn conversations from canonical cases with complete metadata.
"""

import argparse
import json
import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from pathlib import Path

from .canonical_cases import CanonicalCase, CaseType
from .policy_engine import PolicyEngine, PolicyMatch, Decision


# Response templates by decision
RESPONSE_TEMPLATES = {
    Decision.FULL_REFUND: [
        "根据我们的退货政策，您的订单符合退货条件，可以申请全额退款。请您登录账号，在订单页面点击'申请退货'即可。",
        "好的，您的订单在7天无理由退货范围内，可以申请退货退款。请准备好商品和包装，我们会安排快递上门取件。",
    ],
    Decision.EXCHANGE: [
        "您的换货需求我已了解。请您提供一下期望更换的规格，我来帮您查询库存情况。",
        "符合换货条件，请问您想换成什么颜色或规格的呢？",
    ],
    Decision.REJECT: [
        "非常抱歉，您的订单已超过7天无理由退货期限，且不属于质量问题，无法直接办理退货。但我可以帮您转接人工客服，看是否有其他解决方案。",
        "不好意思，超过无理由退货时间了。不过如果您有质量问题凭证，我们可以按质量问题处理。",
    ],
    Decision.NEED_MORE_INFO: [
        "我理解您的问题，为了更好地帮您处理，请提供一下订单号或者购买时间，方便我查询您的订单信息。",
        "好的，请稍等，请问能告诉我您的订单号吗？这样我可以查询具体情况。",
    ],
    Decision.ESCALATE: [
        "好的，您的问题比较特殊，我为您转接人工客服，请稍等。",
        "了解，我这边帮您转接到专员处理，请不要挂断。",
    ],
    Decision.PROVIDE_INFO: [
        "根据产品描述，这款耳机支持蓝牙5.0，兼容市面上主流的智能手机，包括华为、小米、苹果等品牌。",
        "这款充电器的功率是65W，支持PD快充协议，可以为大多数笔记本电脑充电。",
    ],
    Decision.CANCEL_WITH_COMPENSATION: [
        "非常抱歉给您带来不便，由于我们的原因导致延迟发货，您可以申请取消订单并获得赔偿，或者我们尽快安排发货。",
    ],
    Decision.RESEND_OR_REFUND: [
        "经过核实，物流显示签收但您未收到，这种情况我们可以为您补发或退款，请问您希望怎么处理？",
    ],
    Decision.ACKNOWLEDGE_APOLOGIZE: [
        "非常抱歉给您带来不愉快的购物体验，我理解您的心情。我们会尽快处理您的问题。",
        "对不起让您等待这么久，我现在就帮您查询具体情况。",
    ],
    Decision.REFUND_DIFFERENCE: [
        "经核实，同款商品确实降价了，差价{amount}元会退回到您的支付账户，请注意查收。",
    ],
    Decision.COUPON_NOT_APPLICABLE: [
        "不好意思，您的订单金额为{amount}元，不满足29元的优惠券使用门槛。",
    ],
    Decision.PROVIDE_INFO: [
        "好的，我来帮您查询物流信息。请稍等...",
    ],
}


@dataclass
class SyntheticConversation:
    """A synthetic multi-turn conversation with complete metadata."""
    sample_id: str
    messages: List[Dict[str, str]]
    intent: str
    slots: Dict[str, Any]
    policy_ids: List[str]
    decision: str
    requires_human: bool
    source_id: str = "owned_sop_v1"
    source_type: str = "synthetic_from_owned_sop"
    source_revision: str = ""
    parent_case_id: str = ""
    synthetic: bool = True
    generator_model: str = "deterministic-template-v1"
    generator_revision: str = "1.0"
    prompt_hash: Optional[str] = None
    template_family: str = ""
    pii_status: str = "passed"
    dedup_cluster_id: Optional[str] = None
    quality_status: str = "generated"
    review_status: str = "pending"
    
    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "messages": self.messages,
            "intent": self.intent,
            "slots": self.slots,
            "policy_ids": self.policy_ids,
            "decision": self.decision,
            "requires_human": self.requires_human,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_revision": self.source_revision,
            "parent_case_id": self.parent_case_id,
            "synthetic": self.synthetic,
            "generator_model": self.generator_model,
            "generator_revision": self.generator_revision,
            "prompt_hash": self.prompt_hash,
            "template_family": self.template_family,
            "pii_status": self.pii_status,
            "dedup_cluster_id": self.dedup_cluster_id,
            "quality_status": self.quality_status,
            "review_status": self.review_status,
        }


class DeterministicTemplateGenerator:
    """
    Deterministic template-based conversation generator.
    
    Does not require external API. For offline testing.
    """
    
    SYSTEM_PROMPT = """你是一个专业、热情的3C电子产品店客服，很乐意帮助用户解决问题。
重要原则：
1. 只依据给出的政策信息回答，不要编造
2. 缺少订单信息时先澄清，不要猜测
3. 不确定时建议转人工
4. 保持礼貌和专业"""
    
    GREETINGS = [
        "您好，欢迎光临3C数码旗舰店，请问有什么可以帮您？",
        "您好，请问有什么需要咨询的吗？",
        "亲，您好～有什么问题可以随时问我哦～",
    ]
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def generate(
        self, 
        case: CanonicalCase,
        samples_per_case: int = 3,
    ) -> List[SyntheticConversation]:
        """Generate conversation variants from a canonical case."""
        conversations = []
        
        for i in range(samples_per_case):
            conv = self._generate_single(case, i)
            conversations.append(conv)
        
        return conversations
    
    def _generate_single(
        self, 
        case: CanonicalCase, 
        variant_idx: int
    ) -> SyntheticConversation:
        """Generate a single conversation variant."""
        messages = []
        
        # Add greeting (optional, sometimes skip for more natural flow)
        if variant_idx % 3 != 0:
            greeting = self.GREETINGS[variant_idx % len(self.GREETINGS)]
            messages.append({"role": "assistant", "content": greeting})
        
        # Add case turns
        for turn in case.turns:
            messages.append({"role": turn["role"], "content": turn["content"]})
            
            # Generate assistant response after user turn
            if turn["role"] == "user":
                response = self._generate_response(case)
                messages.append({"role": "assistant", "content": response})
        
        # Generate conversation ID
        sample_id = f"ecom_sft_{case.case_id}_{variant_idx}"
        
        return SyntheticConversation(
            sample_id=sample_id,
            messages=messages,
            intent=case.intent,
            slots=case.context,
            policy_ids=case.expected_policy_ids,
            decision=case.expected_decision,
            requires_human=case.requires_human,
            parent_case_id=case.case_id,
            template_family=f"{case.case_type}_{variant_idx}",
        )
    
    def _generate_response(self, case: CanonicalCase) -> str:
        """Generate assistant response based on case."""
        # Find matching decision
        try:
            decision = Decision(case.expected_decision)
        except ValueError:
            decision = Decision.NEED_MORE_INFO
        
        # Get templates for this decision
        templates = RESPONSE_TEMPLATES.get(decision, RESPONSE_TEMPLATES[Decision.NEED_MORE_INFO])
        
        # Pick one deterministically
        template_idx = hash(f"{case.case_id}_{case.expected_decision}") % len(templates)
        response = templates[template_idx]
        
        # Handle missing slots clarification
        if case.expected_decision == "need_more_info" and case.expected_missing_slots:
            missing = case.expected_missing_slots[0]
            if missing == "order_id":
                response = "我理解您的问题，请提供一下订单号方便我查询。"
            elif missing == "days_since_delivery":
                response = "请问您的订单是什么时候签收的呢？"
            elif missing == "package_status":
                response = "请问商品包装和商品本身的状况如何？"
        
        # Handle escalation
        if case.requires_human:
            response = "好的，您的问题我需要转接专员处理，请稍等。"
        
        return response


class LLMGenerator:
    """
    LLM-based conversation generator (optional).
    
    Requires API key to use.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self._available = self.api_key is not None
    
    @property
    def available(self) -> bool:
        """Check if LLM generation is available."""
        return self._available
    
    def generate(self, case: CanonicalCase) -> SyntheticConversation:
        """Generate using LLM."""
        if not self.available:
            raise RuntimeError(
                "LLM generator not available. "
                "Set OPENAI_API_KEY or use DeterministicTemplateGenerator."
            )
        # TODO: Implement LLM-based generation
        raise NotImplementedError("LLM generation not yet implemented")


class ConversationGenerator:
    """
    Main conversation generator.
    
    Supports deterministic template generation (default) and LLM generation (optional).
    """
    
    def __init__(
        self,
        policies: Optional[List[Dict]] = None,
        generator_type: str = "deterministic",
        seed: int = 42,
        source_id: str = "owned_sop_v1",
        source_revision: str = "2026-07-16",
    ):
        self.policies = policies
        self.source_id = source_id
        self.source_revision = source_revision
        
        # Initialize policy engine if policies provided
        self.policy_engine = None
        if policies:
            self.policy_engine = PolicyEngine(policies)
        
        # Initialize generators
        self.seed = seed
        if generator_type == "deterministic":
            self.generator = DeterministicTemplateGenerator(seed=seed)
        elif generator_type == "llm":
            self.generator = LLMGenerator()
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        self.conversations: List[SyntheticConversation] = []
        self._sample_counter = 0
    
    def generate_from_cases(
        self,
        cases: List[CanonicalCase],
        samples_per_case: int = 3,
    ) -> List[SyntheticConversation]:
        """Generate conversations from canonical cases."""
        conversations = []
        
        for case in cases:
            convs = self.generator.generate(case, samples_per_case)
            
            for conv in convs:
                # Validate with policy engine if available
                if self.policy_engine:
                    validation = self._validate_conversation(conv)
                    if not validation["valid"]:
                        conv.quality_status = "quarantine"
                
                # Update metadata
                conv.source_id = self.source_id
                conv.source_revision = self.source_revision
                
                conversations.append(conv)
                self._sample_counter += 1
        
        self.conversations = conversations
        return conversations
    
    def _validate_conversation(self, conv: SyntheticConversation) -> Dict[str, Any]:
        """Validate conversation against policy engine."""
        if not self.policy_engine:
            return {"valid": True}
        
        # Get context from conversation
        user_messages = [
            m["content"] for m in conv.messages 
            if m["role"] == "user"
        ]
        
        # Use last user message as query
        if not user_messages:
            return {"valid": False, "reason": "no_user_message"}
        
        # Match against policy
        match = self.policy_engine.match(conv.slots)
        
        # Check if decision matches
        expected = conv.decision
        actual = match.decision.value
        
        if expected != actual:
            return {
                "valid": False,
                "reason": f"decision_mismatch: expected={expected}, actual={actual}",
                "match": match,
            }
        
        return {"valid": True, "match": match}
    
    def save_conversations(self, output_path: str):
        """Save conversations to JSONL file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in self.conversations:
                f.write(json.dumps(conv.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"Saved {len(self.conversations)} conversations to {output_path}")


def generate_conversations(
    policies_path: str = "data/processed/fixtures/policies.json",
    cases_path: str = "data/fixtures/ecommerce/canonical_cases.jsonl",
    output_path: str = "data/processed/fixtures/conversations.jsonl",
    samples_per_case: int = 3,
    seed: int = 42,
):
    """Main generation function."""
    # Load policies
    with open(policies_path, 'r', encoding='utf-8') as f:
        policies = json.load(f)
    
    # Load cases
    cases = []
    with open(cases_path, 'r', encoding='utf-8') as f:
        for line in f:
            cases.append(CanonicalCase.from_dict(json.loads(line)))
    
    # Generate
    generator = ConversationGenerator(
        policies=policies,
        seed=seed,
    )
    conversations = generator.generate_from_cases(cases, samples_per_case)
    generator.save_conversations(output_path)
    
    print(f"\nGenerated {len(conversations)} conversations")
    
    # Stats
    by_intent = {}
    for conv in conversations:
        by_intent[conv.intent] = by_intent.get(conv.intent, 0) + 1
    
    print("\nBy intent:")
    for intent, count in sorted(by_intent.items()):
        print(f"  - {intent}: {count}")
    
    return conversations


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversations from canonical cases"
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="data/processed/fixtures/policies.json",
        help="Path to policies JSON"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="data/fixtures/ecommerce/canonical_cases.jsonl",
        help="Path to canonical cases JSONL"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed/fixtures/conversations.jsonl",
        help="Output path for conversations JSONL"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples per case"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    generate_conversations(
        policies_path=args.policies,
        cases_path=args.cases,
        output_path=args.output,
        samples_per_case=args.samples,
        seed=args.seed,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
