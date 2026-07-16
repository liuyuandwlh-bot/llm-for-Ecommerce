"""
Canonical Case and Synthetic Conversation Generator

Generates test cases and synthetic conversations based on fictional SOPs.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import json
import random


class CaseType(str, Enum):
    """Canonical case types."""
    NORMAL = "normal"  # 正常情况
    MISSING_SLOT = "missing_slot"  # 缺槽位
    EDGE_CASE = "edge_case"  # 边界条件
    CONFLICT = "conflict"  # 冲突信息
    EMOTIONAL = "emotional"  # 情绪化
    MULTI_TURN = "multi_turn"  # 多轮追问
    MUST_ESCALATE = "must_escalate"  # 必须转人工
    INJECTION = "injection"  # 提示注入


@dataclass
class CanonicalCase:
    """A canonical test case."""
    case_id: str
    intent: str
    case_type: CaseType
    user_query: str
    context: dict
    expected_intent: str
    required_slots: dict
    expected_policy_id: str
    expected_decision: str
    requires_human: bool
    escalation_reason: Optional[str] = None


@dataclass
class SyntheticConversation:
    """A synthetic multi-turn conversation."""
    conv_id: str
    intent: str
    case_id: str
    messages: list[dict]
    policy_ids: list[str]
    source_type: str = "synthetic"
    template_family: str = ""
    quality_status: str = "generated"


class CanonicalCaseGenerator:
    """Generate canonical test cases from policies."""

    def __init__(self):
        self.cases: list[CanonicalCase] = []
        self._case_counter = 0

    def _make_case_id(self) -> str:
        self._case_counter += 1
        return f"case_{self._case_counter:04d}"

    def generate_return_cases(self) -> list[CanonicalCase]:
        """Generate return policy test cases."""
        cases = [
            # Normal cases
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="return_query",
                case_type=CaseType.NORMAL,
                user_query="耳机收到三天了，还没拆封，能退货吗？",
                context={"days_since_delivery": 3, "package_status": "unopened"},
                expected_intent="return_query",
                required_slots={"days_since_delivery": 3, "package_status": "unopened"},
                expected_policy_id="return_3c_001",
                expected_decision="full_refund",
                requires_human=False
            ),
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="return_query",
                case_type=CaseType.NORMAL,
                user_query="耳机拆了但没损坏，能退吗？",
                context={"days_since_delivery": 2, "package_status": "opened", "user_damage": False, "accessories_complete": True},
                expected_intent="return_query",
                required_slots={"days_since_delivery": 2, "package_status": "opened", "user_damage": False},
                expected_policy_id="return_3c_002",
                expected_decision="full_refund",
                requires_human=False
            ),
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="return_query",
                case_type=CaseType.EDGE_CASE,
                user_query="耳机用了10天了，还能退吗？",
                context={"days_since_delivery": 10, "quality_issue": False},
                expected_intent="return_query",
                required_slots={"days_since_delivery": 10},
                expected_policy_id="return_3c_004",
                expected_decision="reject",
                requires_human=False
            ),

            # Missing slot cases
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="return_query",
                case_type=CaseType.MISSING_SLOT,
                user_query="这个充电器能退吗？",
                context={"product_name": "充电器"},
                expected_intent="return_query",
                required_slots={"days_since_delivery": None, "package_status": None},
                expected_policy_id="",
                expected_decision="need_more_info",
                requires_human=False
            ),

            # Quality issue
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="return_query",
                case_type=CaseType.NORMAL,
                user_query="耳机有杂音，听了一周了，是质量问题吗？",
                context={"days_since_delivery": 7, "quality_issue": True, "has_proof": True},
                expected_intent="return_query",
                required_slots={"quality_issue": True, "has_proof": True},
                expected_policy_id="return_3c_003",
                expected_decision="full_refund",
                requires_human=False
            ),
        ]
        self.cases.extend(cases)
        return cases

    def generate_exchange_cases(self) -> list[CanonicalCase]:
        """Generate exchange policy test cases."""
        cases = [
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="exchange_query",
                case_type=CaseType.NORMAL,
                user_query="耳机想换一个白色的，还有吗？",
                context={"exchange_type": "color", "days_since_delivery": 5},
                expected_intent="exchange_query",
                required_slots={"exchange_type": "color", "preferred_color": "白色"},
                expected_policy_id="exchange_3c_001",
                expected_decision="exchange",
                requires_human=False
            ),
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="exchange_query",
                case_type=CaseType.NORMAL,
                user_query="尺码买小了，能换大一号的吗？",
                context={"exchange_type": "size", "days_since_delivery": 10},
                expected_intent="exchange_query",
                required_slots={"exchange_type": "size", "preferred_size": None},
                expected_policy_id="exchange_3c_001",
                expected_decision="exchange",
                requires_human=False
            ),
        ]
        self.cases.extend(cases)
        return cases

    def generate_logistics_cases(self) -> list[CanonicalCase]:
        """Generate logistics test cases."""
        cases = [
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="logistics_query",
                case_type=CaseType.NORMAL,
                user_query="订单号12345，现在到哪了？",
                context={"order_id": "12345"},
                expected_intent="logistics_query",
                required_slots={"order_id": "12345"},
                expected_policy_id="logistics_3c_001",
                expected_decision="provide_tracking",
                requires_human=False
            ),
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="logistics_exception",
                case_type=CaseType.NORMAL,
                user_query="快递显示签收了，但我没收到",
                context={"logistics_status": "signed", "user_received": False},
                expected_intent="logistics_exception",
                required_slots={"logistics_status": "signed", "user_received": False},
                expected_policy_id="logistics_3c_003",
                expected_decision="resend_or_refund",
                requires_human=True
            ),
        ]
        self.cases.extend(cases)
        return cases

    def generate_complaint_cases(self) -> list[CanonicalCase]:
        """Generate complaint test cases."""
        cases = [
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="complaint",
                case_type=CaseType.EMOTIONAL,
                user_query="等了一周还没到，你们怎么回事！太气人了！",
                context={"user_emotion": "angry"},
                expected_intent="complaint",
                required_slots={"user_emotion": "angry"},
                expected_policy_id="complaint_3c_001",
                expected_decision="acknowledge_apologize",
                requires_human=False
            ),
            CanonicalCase(
                case_id=self._make_case_id(),
                intent="escalate",
                case_type=CaseType.MUST_ESCALATE,
                user_query="我要找你们经理投诉！",
                context={},
                expected_intent="escalate",
                required_slots={},
                expected_policy_id="",
                expected_decision="escalate",
                requires_human=True,
                escalation_reason="客户明确要求转管理人员"
            ),
        ]
        self.cases.extend(cases)
        return cases

    def generate_all(self) -> list[CanonicalCase]:
        """Generate all canonical cases."""
        self.generate_return_cases()
        self.generate_exchange_cases()
        self.generate_logistics_cases()
        self.generate_complaint_cases()
        return self.cases

    def save_cases(self, output_path: str):
        """Save cases to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([case.__dict__ for case in self.cases], f, ensure_ascii=False, indent=2)


class ConversationGenerator:
    """Generate synthetic multi-turn conversations."""

    def __init__(self, policies: list[dict]):
        self.policies = policies
        self.conversations: list[SyntheticConversation] = []
        self._conv_counter = 0

    def _make_conv_id(self) -> str:
        self._conv_counter += 1
        return f"conv_{self._conv_counter:05d}"

    def _make_messages(self, case: CanonicalCase) -> list[dict]:
        """Generate multi-turn conversation from canonical case."""
        messages = []

        # User initial query
        messages.append({
            "role": "user",
            "content": case.user_query
        })

        # Agent response (simplified template)
        agent_response = self._generate_agent_response(case)
        messages.append({
            "role": "assistant",
            "content": agent_response
        })

        return messages

    def _generate_agent_response(self, case: CanonicalCase) -> str:
        """Generate agent response based on case type."""
        if case.requires_human:
            return "好的，您的问题比较特殊，我为您转接人工客服，请稍等。"

        if case.expected_decision == "full_refund":
            return "根据我们的退货政策，您的订单符合退货条件，可以申请全额退款。请您登录账号，在订单页面点击'申请退货'即可。"
        elif case.expected_decision == "exchange":
            return "您的换货需求我已了解。请您提供一下期望更换的尺码或颜色，我来帮您查询库存情况。"
        elif case.expected_decision == "reject":
            return "非常抱歉，您的订单已超过7天无理由退货期限，且不属于质量问题，无法直接办理退货。但我可以帮您转接人工客服，看是否有其他解决方案。"
        elif case.expected_decision == "provide_tracking":
            return "好的，我来帮您查询物流信息。请稍等..."
        else:
            return "我理解您的问题，让我帮您查一下相关信息。"

    def generate_from_cases(self, cases: list[CanonicalCase], samples_per_case: int = 3) -> list[SyntheticConversation]:
        """Generate multiple conversation variants from each case."""
        templates = [
            "您好，请问有什么可以帮您？",
            "您好，欢迎光临，有什么需要咨询的吗？",
            "亲，您好～有什么问题可以随时问我哦～",
        ]

        for case in cases:
            for i in range(samples_per_case):
                conv = SyntheticConversation(
                    conv_id=self._make_conv_id(),
                    intent=case.intent,
                    case_id=case.case_id,
                    messages=[
                        {"role": "user", "content": templates[i % len(templates)]},
                        {"role": "assistant", "content": "您好，我是智能客服，很高兴为您服务。请问有什么可以帮助您的？"},
                        {"role": "user", "content": case.user_query},
                        {"role": "assistant", "content": self._generate_agent_response(case)},
                    ],
                    policy_ids=[case.expected_policy_id] if case.expected_policy_id else [],
                    source_type="synthetic_from_case",
                    template_family=f"{case.case_type.value}_{i}",
                    quality_status="generated"
                )
                self.conversations.append(conv)

        return self.conversations

    def save_conversations(self, output_path: str):
        """Save conversations to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in self.conversations:
                f.write(json.dumps(conv.__dict__, ensure_ascii=False) + '\n')


def generate_test_data():
    """Generate test data for the customer service module."""
    from .sop_builder import SOPBuilder

    # Build SOPs
    sop_builder = SOPBuilder()
    policies = sop_builder.build_fictional_store_sops()
    policy_dicts = [p.to_dict() for p in policies]

    # Generate canonical cases
    case_generator = CanonicalCaseGenerator()
    cases = case_generator.generate_all()
    case_generator.save_cases("data/processed/canonical_cases.json")
    print(f"Generated {len(cases)} canonical cases")

    # Generate synthetic conversations
    conv_generator = ConversationGenerator(policy_dicts)
    conversations = conv_generator.generate_from_cases(cases, samples_per_case=2)
    conv_generator.save_conversations("data/processed/synthetic_conversations.jsonl")
    print(f"Generated {len(conversations)} synthetic conversations")


if __name__ == "__main__":
    generate_test_data()
