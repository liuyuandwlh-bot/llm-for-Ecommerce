"""
SOP (Standard Operating Procedures) Builder for Fictional 3C Store

This module defines the policy decision tables and generates machine-readable
fictional policies for the e-commerce customer service domain.

Business Scope:
- 3C Electronics Store (headphones, chargers, cables, phone cases)
- 6 Intent Categories: logistics, returns, exchanges, specifications, coupons, complaints

All policy IDs are explicitly defined and stable, not auto-generated counters.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# Policy Decision Categories
ReturnDecision = Literal["full_refund", "exchange", "manual_inspection", "reject"]
ExchangeDecision = Literal["exchange", "repair", "reject"]
LogisticsDecision = Literal[
    "ship_within_48h", "cancel_with_compensation", "resend_or_refund", "need_more_info"
]


@dataclass
class PolicyCondition:
    """A single condition for a policy decision."""

    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in
    value: Any
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

    policy_id: str  # Stable, explicitly defined
    version: str
    effective_from: str
    category: str  # return, exchange, logistics, specification, coupon, complaint
    title: str
    description: str
    conditions: list[PolicyCondition]
    decisions: list[PolicyDecision]
    evidence: str  # Reference to SOP document
    owner: str = "project_author"
    created_at: str = ""
    effective_to: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Policy":
        """Create from dict."""
        data = dict(data)
        data["conditions"] = [PolicyCondition(**c) for c in data.get("conditions", [])]
        data["decisions"] = [PolicyDecision(**d) for d in data.get("decisions", [])]
        return cls(**data)

    def check_conditions(self, context: dict) -> PolicyDecision | None:
        """Check if context matches policy conditions."""
        for condition in self.conditions:
            if not self._check_single_condition(context, condition):
                return None
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
    """
    Builder for fictional 3C store SOP policies.

    Uses explicitly defined stable policy IDs.
    """

    def __init__(self, version: str = "2026-07-16"):
        self.policies: list[Policy] = []
        self.version = version
        self.created_at = ""  # Set when build is called

    def add_policy(
        self,
        policy_id: str,  # Explicitly defined
        category: str,
        title: str,
        description: str,
        conditions: list[dict],
        decisions: list[dict],
        evidence: str = "",
    ) -> Policy:
        """Add a policy with explicit ID."""
        policy = Policy(
            policy_id=policy_id,
            version=self.version,
            effective_from=self.version,
            category=category,
            title=title,
            description=description,
            conditions=[PolicyCondition(**c) for c in conditions],
            decisions=[PolicyDecision(**d) for d in decisions],
            evidence=evidence or f"虚构商家{category}SOP",
            created_at=self.created_at,
        )
        self.policies.append(policy)
        return policy

    # === RETURN POLICIES ===

    def add_return_policy(
        self,
        policy_id: str,
        title: str,
        description: str,
        conditions: list[dict],
        decisions: list[dict],
        evidence: str = "",
    ) -> Policy:
        """Add a return policy."""
        return self.add_policy(
            policy_id=policy_id,
            category="return",
            title=title,
            description=description,
            conditions=conditions,
            decisions=decisions,
            evidence=evidence,
        )

    def build_fictional_store_sops(self) -> list[Policy]:
        """Build complete fictional 3C store SOPs with stable IDs."""

        # === RETURN POLICIES ===

        # return_001: 7天无理由退货（未拆封）
        self.add_return_policy(
            policy_id="return_001",
            title="7天无理由退货（未拆封）",
            description="商品未拆封且不影响二次销售，可在7天内申请退货",
            conditions=[
                {
                    "field": "days_since_delivery",
                    "operator": "lte",
                    "value": 7,
                    "description": "收货7天内",
                },
                {
                    "field": "package_status",
                    "operator": "eq",
                    "value": "unopened",
                    "description": "包装未拆封",
                },
                {
                    "field": "user_damage",
                    "operator": "eq",
                    "value": False,
                    "description": "无人为损坏",
                },
            ],
            decisions=[
                {
                    "decision": "full_refund",
                    "reasoning": "符合7天无理由退货条件",
                    "requires_human": False,
                }
            ],
            evidence="虚构商家售后SOP 第3.1条",
        )

        # return_002: 7天无理由退货（已拆封完好）
        self.add_return_policy(
            policy_id="return_002",
            title="7天无理由退货（已拆封完好）",
            description="已拆封但商品完好、配件齐全，可在7天内申请退货",
            conditions=[
                {
                    "field": "days_since_delivery",
                    "operator": "lte",
                    "value": 7,
                    "description": "收货7天内",
                },
                {
                    "field": "package_status",
                    "operator": "eq",
                    "value": "opened",
                    "description": "包装已拆封",
                },
                {
                    "field": "user_damage",
                    "operator": "eq",
                    "value": False,
                    "description": "无人为损坏",
                },
                {
                    "field": "accessories_complete",
                    "operator": "eq",
                    "value": True,
                    "description": "配件齐全",
                },
            ],
            decisions=[
                {
                    "decision": "full_refund",
                    "reasoning": "已拆封但商品完好，符合退货条件",
                    "requires_human": False,
                }
            ],
            evidence="虚构商家售后SOP 第3.2条",
        )

        # return_003: 质量问题退货
        self.add_return_policy(
            policy_id="return_003",
            title="质量问题退货",
            description="商品存在质量问题，不受7天限制，可申请退货",
            conditions=[
                {
                    "field": "quality_issue",
                    "operator": "eq",
                    "value": True,
                    "description": "确认存在质量问题",
                },
                {
                    "field": "has_proof",
                    "operator": "eq",
                    "value": True,
                    "description": "有质量问题凭证",
                },
            ],
            decisions=[
                {
                    "decision": "full_refund",
                    "reasoning": "质量问题可申请退货退款",
                    "requires_human": False,
                },
                {
                    "decision": "manual_inspection",
                    "reasoning": "需要人工核实质量问题",
                    "requires_human": True,
                },
            ],
            evidence="虚构商家售后SOP 第3.3条",
        )

        # return_004: 超期退货申请
        self.add_return_policy(
            policy_id="return_004",
            title="超期退货申请",
            description="超过7天无理由退货期，不适用无理由退货",
            conditions=[
                {
                    "field": "days_since_delivery",
                    "operator": "gt",
                    "value": 7,
                    "description": "超过7天",
                },
                {
                    "field": "quality_issue",
                    "operator": "eq",
                    "value": False,
                    "description": "非质量问题",
                },
            ],
            decisions=[
                {
                    "decision": "reject",
                    "reasoning": "超过无理由退货期限",
                    "requires_human": False,
                    "escalation_reasons": ["用户强烈投诉可转人工"],
                }
            ],
            evidence="虚构商家售后SOP 第3.4条",
        )

        # === EXCHANGE POLICIES ===

        # exchange_001: 换货-颜色（3C 业务，去掉服装尺码）
        self.add_policy(
            policy_id="exchange_001",
            category="exchange",
            title="换货-颜色或型号",
            description="同款商品可换颜色或型号，需商品完好",
            conditions=[
                {
                    "field": "days_since_delivery",
                    "operator": "lte",
                    "value": 15,
                    "description": "收货15天内",
                },
                {
                    "field": "user_damage",
                    "operator": "eq",
                    "value": False,
                    "description": "无人为损坏",
                },
                {
                    "field": "exchange_type",
                    "operator": "in",
                    "value": ["color", "variant"],
                    "description": "换同款颜色或型号",
                },
            ],
            decisions=[
                {"decision": "exchange", "reasoning": "符合换货条件", "requires_human": False}
            ],
            evidence="虚构商家售后SOP 第4.1条",
        )

        # exchange_002: 质量问题换货
        self.add_policy(
            policy_id="exchange_002",
            category="exchange",
            title="质量问题换货",
            description="质量问题可申请换货或维修",
            conditions=[
                {
                    "field": "quality_issue",
                    "operator": "eq",
                    "value": True,
                    "description": "确认质量问题",
                },
            ],
            decisions=[
                {"decision": "exchange", "reasoning": "质量问题换货", "requires_human": False},
                {"decision": "repair", "reasoning": "如无库存可申请维修", "requires_human": True},
            ],
            evidence="虚构商家售后SOP 第4.2条",
        )

        # === LOGISTICS POLICIES ===

        # logistics_001: 发货时间规定 — 现货且未发货且在48h 承诺期内则 ship_within_48h
        self.add_policy(
            policy_id="logistics_001",
            category="logistics",
            title="发货时间规定",
            description="现货订单在48h承诺期内未发货，承诺48h内发货",
            conditions=[
                {
                    "field": "order_type",
                    "operator": "eq",
                    "value": "in_stock",
                    "description": "现货订单",
                },
                {"field": "shipped", "operator": "eq", "value": False, "description": "未发货"},
                {
                    "field": "days_since_order",
                    "operator": "lte",
                    "value": 2,
                    "description": "距下单 48h 以内",
                },
            ],
            decisions=[
                {
                    "decision": "ship_within_48h",
                    "reasoning": "现货订单48小时内发货",
                    "requires_human": False,
                }
            ],
            evidence="虚构商家物流SOP 第2.1条",
        )

        # logistics_002: 延迟发货处理
        self.add_policy(
            policy_id="logistics_002",
            category="logistics",
            title="延迟发货处理",
            description="超过承诺发货时间未发货，可申请赔偿或取消",
            conditions=[
                {
                    "field": "order_type",
                    "operator": "eq",
                    "value": "in_stock",
                    "description": "现货订单",
                },
                {"field": "shipped", "operator": "eq", "value": False, "description": "未发货"},
                {
                    "field": "days_since_order",
                    "operator": "gt",
                    "value": 2,
                    "description": "付款超过48小时",
                },
            ],
            decisions=[
                {
                    "decision": "cancel_with_compensation",
                    "reasoning": "延迟发货可赔偿或取消",
                    "requires_human": True,
                }
            ],
            evidence="虚构商家物流SOP 第2.2条",
        )

        # logistics_003: 丢件处理
        self.add_policy(
            policy_id="logistics_003",
            category="logistics",
            title="丢件处理",
            description="物流显示签收但用户未收到，可申请退款或补发",
            conditions=[
                {
                    "field": "logistics_status",
                    "operator": "eq",
                    "value": "signed",
                    "description": "物流显示已签收",
                },
                {
                    "field": "user_received",
                    "operator": "eq",
                    "value": False,
                    "description": "用户确认未收到",
                },
            ],
            decisions=[
                {
                    "decision": "resend_or_refund",
                    "reasoning": "丢件可补发或退款",
                    "requires_human": True,
                }
            ],
            evidence="虚构商家物流SOP 第2.3条",
        )

        # === COUPON/POLICY PROTECTION POLICIES ===

        # coupon_001: 价格保护政策
        self.add_policy(
            policy_id="coupon_001",
            category="coupon",
            title="价格保护政策",
            description="订单签收后7天内，同款商品降价可申请差价退还",
            conditions=[
                {
                    "field": "days_since_signed",
                    "operator": "lte",
                    "value": 7,
                    "description": "签收7天内",
                },
                {
                    "field": "price_dropped",
                    "operator": "eq",
                    "value": True,
                    "description": "价格确实下降",
                },
                {
                    "field": "same_product",
                    "operator": "eq",
                    "value": True,
                    "description": "同款商品",
                },
            ],
            decisions=[
                {
                    "decision": "refund_difference",
                    "reasoning": "符合价格保护条件",
                    "requires_human": False,
                }
            ],
            evidence="虚构商家优惠SOP 第5.1条",
        )

        # coupon_002: 优惠券使用限制
        self.add_policy(
            policy_id="coupon_002",
            category="coupon",
            title="优惠券使用限制",
            description="部分情况不可使用优惠券",
            conditions=[
                {
                    "field": "order_amount",
                    "operator": "lt",
                    "value": 29,
                    "description": "订单金额低于29元",
                },
            ],
            decisions=[
                {
                    "decision": "coupon_not_applicable",
                    "reasoning": "订单金额不满足优惠券使用条件",
                    "requires_human": False,
                }
            ],
            evidence="虚构商家优惠SOP 第5.2条",
        )

        # === SPECIFICATION POLICIES ===

        # specification_001: 商品兼容性查询 — 需要 target_device 才能精确回答
        self.add_policy(
            policy_id="specification_001",
            category="specification",
            title="商品兼容性查询",
            description="提供商品兼容性信息，帮助用户确认是否适用",
            conditions=[
                {
                    "field": "user_asked_compatibility",
                    "operator": "eq",
                    "value": True,
                    "description": "询问兼容性问题",
                },
                {
                    "field": "target_device",
                    "operator": "in",
                    "value": ["华为P50", "iPhone 15", "小米14"],
                    "description": "已知设备",
                },
            ],
            decisions=[
                {"decision": "provide_info", "reasoning": "提供兼容性信息", "requires_human": False}
            ],
            evidence="虚构商家商品SOP 第6.1条",
        )

        # === COMPLAINT POLICIES ===

        # complaint_001: 服务投诉处理
        self.add_policy(
            policy_id="complaint_001",
            category="complaint",
            title="服务投诉处理",
            description="用户对服务态度不满，可升级处理",
            conditions=[
                {
                    "field": "complaint_type",
                    "operator": "eq",
                    "value": "service_attitude",
                    "description": "服务态度投诉",
                },
            ],
            decisions=[
                {
                    "decision": "escalate",
                    "reasoning": "服务投诉需升级主管处理",
                    "requires_human": True,
                }
            ],
            evidence="虚构商家投诉SOP 第7.1条",
        )

        # complaint_002: 虚假宣传投诉
        self.add_policy(
            policy_id="complaint_002",
            category="complaint",
            title="虚假宣传投诉",
            description="如商品描述与实际不符，可退货退款",
            conditions=[
                {
                    "field": "complaint_type",
                    "operator": "eq",
                    "value": "false_advertising",
                    "description": "虚假宣传投诉",
                },
                {
                    "field": "misleading_confirmed",
                    "operator": "eq",
                    "value": True,
                    "description": "确认存在误导",
                },
            ],
            decisions=[
                {
                    "decision": "accept_return_refund",
                    "reasoning": "虚假宣传可退货退款",
                    "requires_human": True,
                }
            ],
            evidence="虚构商家投诉SOP 第7.2条",
        )

        return self.policies

    def save_policies(self, output_path: str):
        """Save policies to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in self.policies], f, ensure_ascii=False, indent=2)

    def generate_policy_report(self) -> dict:
        """Generate statistics about policies."""
        by_category = {}
        for p in self.policies:
            if p.category not in by_category:
                by_category[p.category] = []
            by_category[p.category].append(
                {
                    "policy_id": p.policy_id,
                    "title": p.title,
                    "requires_human": any(d.requires_human for d in p.decisions),
                }
            )

        return {
            "total_policies": len(self.policies),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "policies_requiring_human": sum(
                1 for p in self.policies if any(d.requires_human for d in p.decisions)
            ),
        }


def build_sops(output_path: str = "data/processed/fixtures/policies.json"):
    """Build SOPs and save to file."""

    builder = SOPBuilder()
    # Use a fixed fixture date so that generated policies are
    # reproducible across independent processes.
    builder.created_at = "2026-01-01"
    policies = builder.build_fictional_store_sops()

    builder.save_policies(output_path)

    report = builder.generate_policy_report()
    print(f"Generated {report['total_policies']} policies:")
    for category, count in report["by_category"].items():
        print(f"  - {category}: {count}")
    print(f"Policies requiring human escalation: {report['policies_requiring_human']}")

    return policies


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build SOP policies for fictional 3C store")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/processed/fixtures/policies.json",
        help="Output path for policies JSON",
    )
    parser.add_argument("--validate", action="store_true", help="Validate policies after building")
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Optional data registry JSON. If given, the built artifact's SHA-256 is recorded and the source status is moved to acquired.",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default="owned_sop_v1",
        help="Registry source_id to update on success.",
    )

    args = parser.parse_args()

    # Build SOPs
    policies = build_sops(args.output)

    # Validate if requested
    if args.validate:
        from .policy_engine import validate_policy_uniqueness

        errors = validate_policy_uniqueness([p.to_dict() for p in policies])
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  - {e}")
            return 1
        print("Validation passed: all policy IDs unique")

    if args.registry:
        try:
            import hashlib

            from .registry import update_registry_checksum

            with open(args.output, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            update_registry_checksum(
                args.registry,
                args.source_id,
                checksum=digest,
                status="acquired",
            )
            print(f"Registry updated: {args.source_id} -> {digest[:12]}...")
        except Exception as exc:
            print(f"Registry update failed: {exc}")
            return 1

    print(f"\nOutput saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
