"""
SOP (Standard Operating Procedures) Builder for Fictional 3C Store

This module defines the policy decision tables and generates machine-readable
fictional policies for the e-commerce customer service domain.

Business Scope:
- 3C Electronics Store (headphones, chargers, cables, phone cases)
- 6 Intent Categories: logistics, returns, exchanges, specifications, coupons, complaints
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from datetime import datetime
import json
import hashlib


# Policy Decision Categories
ReturnCategory = Literal["full_refund", "exchange", "manual_inspection", "reject"]
ExchangeCategory = Literal["same_model", "different_model", "repair", "reject"]
LogisticsStatus = Literal["in_transit", "delivered", "exception", "pending"]


@dataclass
class PolicyCondition:
    """A single condition for a policy decision."""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in
    value: any
    description: str = ""


@dataclass
class PolicyDecision:
    """A decision outcome with conditions."""
    decision: str
    reasoning: str
    requires_human: bool = False
    escalation_reasons: list[str] = field(default_factory=list)


@dataclass
class Policy:
    """A complete policy with conditions and decisions."""
    policy_id: str
    version: str
    effective_from: str
    category: str  # return, exchange, logistics, specification, coupon, complaint
    title: str
    description: str
    conditions: list[PolicyCondition]
    decisions: list[PolicyDecision]
    evidence: str  # Reference to SOP document
    owner: str = "project_author"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def check_conditions(self, context: dict) -> Optional[PolicyDecision]:
        """Check if context matches policy conditions."""
        for condition in self.conditions:
            if not self._check_single_condition(context, condition):
                return None
        # Return the first matching decision
        return self.decisions[0] if self.decisions else None

    def _check_single_condition(self, context: dict, condition: PolicyCondition) -> bool:
        """Check a single condition against context."""
        value = context.get(condition.field)
        if value is None:
            return False

        target = condition.value
        op = condition.operator

        if op == "eq":
            return value == target
        elif op == "ne":
            return value != target
        elif op == "gt":
            return value > target
        elif op == "lt":
            return value < target
        elif op == "gte":
            return value >= target
        elif op == "lte":
            return value <= target
        elif op == "in":
            return value in target
        elif op == "not_in":
            return value not in target
        return False


class SOPBuilder:
    """Builder for fictional 3C store SOP policies."""

    def __init__(self):
        self.policies: list[Policy] = []
        self._policy_counter = 0

    def _make_policy_id(self, category: str) -> str:
        """Generate a unique policy ID."""
        self._policy_counter += 1
        return f"{category}_3c_{self._policy_counter:03d}"

    def add_return_policy(
        self,
        title: str,
        description: str,
        conditions: list[dict],
        decisions: list[dict],
        evidence: str = ""
    ) -> Policy:
        """Add a return policy."""
        policy = Policy(
            policy_id=self._make_policy_id("return"),
            version="2026-07-16",
            effective_from="2026-07-16",
            category="return",
            title=title,
            description=description,
            conditions=[PolicyCondition(**c) for c in conditions],
            decisions=[PolicyDecision(**d) for d in decisions],
            evidence=evidence or "虚构商家售后SOP"
        )
        self.policies.append(policy)
        return policy

    def add_exchange_policy(
        self,
        title: str,
        description: str,
        conditions: list[dict],
        decisions: list[dict],
        evidence: str = ""
    ) -> Policy:
        """Add an exchange policy."""
        policy = Policy(
            policy_id=self._make_policy_id("exchange"),
            version="2026-07-16",
            effective_from="2026-07-16",
            category="exchange",
            title=title,
            description=description,
            conditions=[PolicyCondition(**c) for c in conditions],
            decisions=[PolicyDecision(**d) for d in decisions],
            evidence=evidence or "虚构商家售后SOP"
        )
        self.policies.append(policy)
        return policy

    def add_logistics_policy(
        self,
        title: str,
        description: str,
        conditions: list[dict],
        decisions: list[dict],
        evidence: str = ""
    ) -> Policy:
        """Add a logistics policy."""
        policy = Policy(
            policy_id=self._make_policy_id("logistics"),
            version="2026-07-16",
            effective_from="2026-07-16",
            category="logistics",
            title=title,
            description=description,
            conditions=[PolicyCondition(**c) for c in conditions],
            decisions=[PolicyDecision(**d) for d in decisions],
            evidence=evidence or "虚构商家物流SOP"
        )
        self.policies.append(policy)
        return policy

    def add_coupon_policy(
        self,
        title: str,
        description: str,
        conditions: list[dict],
        decisions: list[dict],
        evidence: str = ""
    ) -> Policy:
        """Add a coupon/price protection policy."""
        policy = Policy(
            policy_id=self._make_policy_id("coupon"),
            version="2026-07-16",
            effective_from="2026-07-16",
            category="coupon",
            title=title,
            description=description,
            conditions=[PolicyCondition(**c) for c in conditions],
            decisions=[PolicyDecision(**d) for d in decisions],
            evidence=evidence or "虚构商家优惠SOP"
        )
        self.policies.append(policy)
        return policy

    def add_specification_policy(
        self,
        title: str,
        description: str,
        conditions: list[dict],
        decisions: list[dict],
        evidence: str = ""
    ) -> Policy:
        """Add a product specification policy."""
        policy = Policy(
            policy_id=self._make_policy_id("specification"),
            version="2026-07-16",
            effective_from="2026-07-16",
            category="specification",
            title=title,
            description=description,
            conditions=[PolicyCondition(**c) for c in conditions],
            decisions=[PolicyDecision(**d) for d in decisions],
            evidence=evidence or "虚构商家商品SOP"
        )
        self.policies.append(policy)
        return policy

    def add_complaint_policy(
        self,
        title: str,
        description: str,
        conditions: list[dict],
        decisions: list[dict],
        evidence: str = ""
    ) -> Policy:
        """Add a complaint handling policy."""
        policy = Policy(
            policy_id=self._make_policy_id("complaint"),
            version="2026-07-16",
            effective_from="2026-07-16",
            category="complaint",
            title=title,
            description=description,
            conditions=[PolicyCondition(**c) for c in conditions],
            decisions=[PolicyDecision(**d) for d in decisions],
            evidence=evidence or "虚构商家投诉SOP"
        )
        self.policies.append(policy)
        return policy

    def build_fictional_store_sops(self) -> list[Policy]:
        """Build complete fictional 3C store SOPs."""

        # === RETURN POLICIES ===

        # 退货-未拆封
        self.add_return_policy(
            title="7天无理由退货（未拆封）",
            description="商品未拆封且不影响二次销售，可在7天内申请退货",
            conditions=[
                {"field": "days_since_delivery", "operator": "lte", "value": 7, "description": "收货7天内"},
                {"field": "package_status", "operator": "eq", "value": "unopened", "description": "包装未拆封"},
                {"field": "damage_by_user", "operator": "eq", "value": False, "description": "无人为损坏"},
            ],
            decisions=[
                {"decision": "full_refund", "reasoning": "符合7天无理由退货条件", "requires_human": False}
            ],
            evidence="虚构商家售后SOP 第3.1条"
        )

        # 退货-已拆封但完好
        self.add_return_policy(
            title="7天无理由退货（已拆封完好）",
            description="已拆封但商品完好、配件齐全，可在7天内申请退货",
            conditions=[
                {"field": "days_since_delivery", "operator": "lte", "value": 7, "description": "收货7天内"},
                {"field": "package_status", "operator": "eq", "value": "opened", "description": "包装已拆封"},
                {"field": "user_damage", "operator": "eq", "value": False, "description": "无人为损坏"},
                {"field": "accessories_complete", "operator": "eq", "value": True, "description": "配件齐全"},
            ],
            decisions=[
                {"decision": "full_refund", "reasoning": "已拆封但商品完好，符合退货条件", "requires_human": False}
            ],
            evidence="虚构商家售后SOP 第3.2条"
        )

        # 退货-质量问题
        self.add_return_policy(
            title="质量问题退货",
            description="商品存在质量问题，不受7天限制，可申请退货",
            conditions=[
                {"field": "quality_issue", "operator": "eq", "value": True, "description": "确认存在质量问题"},
                {"field": "has_proof", "operator": "eq", "value": True, "description": "有质量问题凭证"},
            ],
            decisions=[
                {"decision": "full_refund", "reasoning": "质量问题可申请退货退款", "requires_human": False},
                {"decision": "manual_inspection", "reasoning": "需要人工核实质量问题", "requires_human": True}
            ],
            evidence="虚构商家售后SOP 第3.3条"
        )

        # 退货-超期
        self.add_return_policy(
            title="超期退货申请",
            description="超过7天无理由退货期，不适用无理由退货",
            conditions=[
                {"field": "days_since_delivery", "operator": "gt", "value": 7, "description": "超过7天"},
                {"field": "quality_issue", "operator": "eq", "value": False, "description": "非质量问题"},
            ],
            decisions=[
                {"decision": "reject", "reasoning": "超过无理由退货期限", "requires_human": False, "escalation_reasons": ["用户强烈投诉"]}
            ],
            evidence="虚构商家售后SOP 第3.4条"
        )

        # === EXCHANGE POLICIES ===

        # 换货-尺码/颜色
        self.add_exchange_policy(
            title="换货-尺码或颜色",
            description="同款商品可换尺码或颜色，需商品完好",
            conditions=[
                {"field": "days_since_delivery", "operator": "lte", "value": 15, "description": "收货15天内"},
                {"field": "user_damage", "operator": "eq", "value": False, "description": "无人为损坏"},
                {"field": "exchange_type", "operator": "in", "value": ["size", "color"], "description": "换同款尺码或颜色"},
            ],
            decisions=[
                {"decision": "exchange", "reasoning": "符合换货条件", "requires_human": False}
            ],
            evidence="虚构商家售后SOP 第4.1条"
        )

        # 换货-质量问题
        self.add_exchange_policy(
            title="质量问题换货",
            description="质量问题可申请换货或维修",
            conditions=[
                {"field": "quality_issue", "operator": "eq", "value": True, "description": "确认质量问题"},
            ],
            decisions=[
                {"decision": "exchange", "reasoning": "质量问题换货", "requires_human": False},
                {"decision": "repair", "reasoning": "如无库存可申请维修", "requires_human": True}
            ],
            evidence="虚构商家售后SOP 第4.2条"
        )

        # === LOGISTICS POLICIES ===

        # 物流-发货时间
        self.add_logistics_policy(
            title="发货时间规定",
            description="现货订单48小时内发货，预售订单按页面时间",
            conditions=[
                {"field": "order_type", "operator": "eq", "value": "in_stock", "description": "现货订单"},
                {"field": "payment_confirmed", "operator": "eq", "value": True, "description": "已付款"},
            ],
            decisions=[
                {"decision": "ship_within_48h", "reasoning": "现货订单48小时内发货", "requires_human": False}
            ],
            evidence="虚构商家物流SOP 第2.1条"
        )

        # 物流-延迟发货
        self.add_logistics_policy(
            title="延迟发货处理",
            description="超过承诺发货时间未发货，可申请赔偿或取消",
            conditions=[
                {"field": "shipped", "operator": "eq", "value": False, "description": "未发货"},
                {"field": "hours_since_承诺时间", "operator": "gt", "value": 0, "description": "已超过承诺时间"},
            ],
            decisions=[
                {"decision": "cancel_with_compensation", "reasoning": "延迟发货可赔偿或取消", "requires_human": True}
            ],
            evidence="虚构商家物流SOP 第2.2条"
        )

        # 物流-丢件
        self.add_logistics_policy(
            title="丢件处理",
            description="物流显示签收但用户未收到，可申请退款或补发",
            conditions=[
                {"field": "logistics_status", "operator": "eq", "value": "signed", "description": "物流显示已签收"},
                {"field": "user_received", "operator": "eq", "value": False, "description": "用户确认未收到"},
            ],
            decisions=[
                {"decision": "resend_or_refund", "reasoning": "丢件可补发或退款", "requires_human": True}
            ],
            evidence="虚构商家物流SOP 第2.3条"
        )

        # === COUPON POLICIES ===

        # 优惠券-价格保护
        self.add_coupon_policy(
            title="价格保护政策",
            description="订单签收后7天内，同款商品降价可申请差价退还",
            conditions=[
                {"field": "days_since_signed", "operator": "lte", "value": 7, "description": "签收7天内"},
                {"field": "price_dropped", "operator": "eq", "value": True, "description": "价格确实下降"},
                {"field": "same_product", "operator": "eq", "value": True, "description": "同款商品"},
            ],
            decisions=[
                {"decision": "refund_difference", "reasoning": "符合价格保护条件", "requires_human": False}
            ],
            evidence="虚构商家优惠SOP 第5.1条"
        )

        # 优惠券-不可用情况
        self.add_coupon_policy(
            title="优惠券使用限制",
            description="部分情况不可使用优惠券",
            conditions=[
                {"field": "order_amount", "operator": "lt", "value": 29, "description": "订单金额低于29元"},
            ],
            decisions=[
                {"decision": "coupon_not_applicable", "reasoning": "订单金额不满足优惠券使用条件", "requires_human": False}
            ],
            evidence="虚构商家优惠SOP 第5.2条"
        )

        # === SPECIFICATION POLICIES ===

        # 规格-兼容性
        self.add_specification_policy(
            title="商品兼容性查询",
            description="提供商品兼容性信息，帮助用户确认是否适用",
            conditions=[
                {"field": "user_asked_compatibility", "operator": "eq", "value": True, "description": "询问兼容性问题"},
            ],
            decisions=[
                {"decision": "provide_compatibility_info", "reasoning": "提供兼容性信息", "requires_human": False}
            ],
            evidence="虚构商家商品SOP 第6.1条"
        )

        # === COMPLAINT POLICIES ===

        # 投诉-服务态度
        self.add_complaint_policy(
            title="服务投诉处理",
            description="用户对服务态度不满，可升级处理",
            conditions=[
                {"field": "complaint_type", "operator": "eq", "value": "service_attitude", "description": "服务态度投诉"},
            ],
            decisions=[
                {"decision": "escalate_to_supervisor", "reasoning": "服务投诉需升级主管处理", "requires_human": True}
            ],
            evidence="虚构商家投诉SOP 第7.1条"
        )

        # 投诉-虚假宣传
        self.add_complaint_policy(
            title="虚假宣传投诉",
            description="如商品描述与实际不符，可退货退款",
            conditions=[
                {"field": "complaint_type", "operator": "eq", "value": "false_advertising", "description": "虚假宣传投诉"},
                {"field": "misleading_confirmed", "operator": "eq", "value": True, "description": "确认存在误导"},
            ],
            decisions=[
                {"decision": "accept_return_refund", "reasoning": "虚假宣传可退货退款", "requires_human": True}
            ],
            evidence="虚构商家投诉SOP 第7.2条"
        )

        return self.policies

    def save_policies(self, output_path: str):
        """Save policies to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in self.policies], f, ensure_ascii=False, indent=2)

    def generate_policy_report(self) -> dict:
        """Generate statistics about policies."""
        by_category = {}
        for p in self.policies:
            if p.category not in by_category:
                by_category[p.category] = []
            by_category[p.category].append({
                "policy_id": p.policy_id,
                "title": p.title,
                "requires_human": any(d.requires_human for d in p.decisions)
            })

        return {
            "total_policies": len(self.policies),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "policies_requiring_human": sum(
                1 for p in self.policies if any(d.requires_human for d in p.decisions)
            )
        }


def build_sops():
    """Build SOPs and save to default location."""
    builder = SOPBuilder()
    policies = builder.build_fictional_store_sops()

    output_path = "data/processed/policies_v1.json"
    builder.save_policies(output_path)

    report = builder.generate_policy_report()
    print(f"Generated {report['total_policies']} policies:")
    for category, count in report['by_category'].items():
        print(f"  - {category}: {count}")
    print(f"Policies requiring human escalation: {report['policies_requiring_human']}")

    return policies


if __name__ == "__main__":
    build_sops()
