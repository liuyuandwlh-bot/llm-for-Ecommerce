"""
Policy Engine for E-commerce Customer Service

Round 2 rewrite:
- Per-policy evaluation of (matched, conflict, missing) conditions
- Explicit category_hint; no fallback to "return"
- Conflict filtering before scoring
- Tied-score returns ambiguous/escalate
- Explicit behavior rules for non-policy intents (out-of-scope, tool-required,
  injection, privacy) via match_behavior()
- SlotSchema.from_dict strict by default; explicit aliases only
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """Policy decision outcomes."""

    FULL_REFUND = "full_refund"
    EXCHANGE = "exchange"
    REPAIR = "repair"
    MANUAL_INSPECTION = "manual_inspection"
    REJECT = "reject"
    NEED_MORE_INFO = "need_more_info"
    ESCALATE = "escalate"
    PROVIDE_INFO = "provide_info"
    COUPON_NOT_APPLICABLE = "coupon_not_applicable"
    REFUND_DIFFERENCE = "refund_difference"
    SHIP_WITHIN_48H = "ship_within_48h"
    CANCEL_WITH_COMPENSATION = "cancel_with_compensation"
    RESEND_OR_REFUND = "resend_or_refund"
    ACKNOWLEDGE_APOLOGIZE = "acknowledge_apologize"
    ACCEPT_RETURN_REFUND = "accept_return_refund"
    AMBIGUOUS = "ambiguous"


# --- Behavior intents that should NOT fall through to the policy-engine default ---


class BehaviorIntent(str, Enum):
    """Non-policy intents routed through behavior rules."""

    OUT_OF_SCOPE = "out_of_scope"
    TOOL_REQUIRED = "tool_required"
    PRIVACY = "privacy"
    INJECTION = "injection"
    ESCALATE = "escalate"


# Mapping from canonical `intent` strings to a behavior dispatch.
# If a case's intent is not policy-backed (return/exchange/logistics/coupon/
# specification/complaint) and looks like one of the above, it dispatches here.
_BEHAVIOR_INTENT_ALIASES: dict[str, BehaviorIntent] = {
    BehaviorIntent.OUT_OF_SCOPE.value: BehaviorIntent.OUT_OF_SCOPE,
    BehaviorIntent.TOOL_REQUIRED.value: BehaviorIntent.TOOL_REQUIRED,
    BehaviorIntent.PRIVACY.value: BehaviorIntent.PRIVACY,
    BehaviorIntent.INJECTION.value: BehaviorIntent.INJECTION,
    BehaviorIntent.ESCALATE.value: BehaviorIntent.ESCALATE,
}


@dataclass
class PolicyMatch:
    """Result of policy matching.

    Attributes:
        policy_id: Empty string when no full match was found.
        decision: The decision enum.
        missing_slots: Conditions that were missing to reach a full match.
        conflicted_policies: Policies whose conditions conflicted with context.
        candidate_policy_ids: All non-conflicting candidates considered.
        ambiguity: True if multiple candidates tied for best score.
        reasoning: Human-readable explanation.
        requires_human: Whether this policy/behavior implies handoff.
        escalation_reasons: Reasons for escalation.
    """

    policy_id: str
    decision: Decision
    reasoning: str
    requires_human: bool = False
    missing_slots: list[str] = field(default_factory=list)
    conflicted_policies: list[str] = field(default_factory=list)
    candidate_policy_ids: list[str] = field(default_factory=list)
    ambiguity: bool = False
    escalation_reasons: list[str] = field(default_factory=list)


@dataclass
class BehaviorResult:
    """Result of matching a non-policy intent through behavior rules."""

    intent: str
    decision: Decision
    requires_human: bool
    reasoning: str
    tool_expectation: str | None = None
    escalation_reasons: list[str] = field(default_factory=list)


# Canonical slot set for the 3C store domain. Aliased fields are explicit and
# only accepted via `aliases` parameter.
_CANONICAL_SLOT_FIELDS: set = {
    # Order-related
    "order_id",
    "order_date",
    "delivery_date",
    "days_since_delivery",
    "days_since_signed",
    "days_since_order",
    # Product
    "product_name",
    "product_category",
    "product_model",
    "product_variant",
    # Package
    "package_status",
    "accessories_complete",
    # Damage / quality
    "user_damage",
    "quality_issue",
    "has_proof",
    "misleading_confirmed",
    # Exchange
    "exchange_type",
    "preferred_color",
    # Logistics
    "logistics_status",
    "shipped",
    "user_received",
    "order_type",
    # Coupon / pricing
    "price_dropped",
    "same_product",
    "order_amount",
    "coupon_code",
    # Complaint
    "complaint_type",
    "user_emotion",
    # Specification
    "user_asked_compatibility",
    "target_device",
}


_SLOT_ALIASES: dict[str, str] = {
    "damage_by_user": "user_damage",  # legacy alias -> canonical
    "preferred_size": "preferred_color",  # legacy 服装 size folded into 3C color
    "size": "product_variant",  # 服装 size -> 3C variant
}


@dataclass
class SlotSchema:
    """Canonical slot definitions for 3C store.

    Only the canonical fields are accepted in `from_dict`. Legacy aliases
    are accepted only when `accept_aliases=True`.
    """

    # Order-related
    order_id: str | None = None
    order_date: str | None = None
    delivery_date: str | None = None
    days_since_delivery: int | None = None
    days_since_signed: int | None = None
    days_since_order: int | None = None

    # Product-related
    product_name: str | None = None
    product_category: str | None = None  # headphones, charger, cable, phone_case
    product_model: str | None = None  # e.g. specific phone case model
    product_variant: str | None = None  # e.g. "iPhone 15 Pro Max" 3C variant

    # Package status
    package_status: str | None = None  # unopened, opened
    accessories_complete: bool | None = None

    # Condition
    user_damage: bool | None = None

    # Quality
    quality_issue: bool | None = None
    has_proof: bool | None = None
    misleading_confirmed: bool | None = None

    # Exchange
    exchange_type: str | None = None  # color, variant
    preferred_color: str | None = None

    # Logistics
    logistics_status: str | None = None  # in_transit, delivered, signed, exception
    shipped: bool | None = None
    user_received: bool | None = None
    order_type: str | None = None  # in_stock, preorder

    # Coupon / pricing
    price_dropped: bool | None = None
    same_product: bool | None = None
    order_amount: float | None = None
    coupon_code: str | None = None

    # Complaint
    complaint_type: str | None = None  # service_attitude, false_advertising
    user_emotion: str | None = None  # calm, angry, frustrated

    # Specification
    user_asked_compatibility: bool | None = None
    target_device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, only including non-None fields."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        accept_aliases: bool = True,
    ) -> "SlotSchema":
        """Create from dict, normalizing legacy aliases.

        By default alias fields like `damage_by_user` are mapped to the
        canonical `user_damage`. Unknown fields raise ValueError instead of
        being silently dropped.
        """
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            canonical = _SLOT_ALIASES.get(key, key)
            if accept_aliases and key != canonical:
                # Inherit only if not already set in input
                canonical = canonical
            if canonical not in _CANONICAL_SLOT_FIELDS:
                raise ValueError(f"Unknown slot field: {key!r}")
            # If both alias and canonical present, canonical wins
            if canonical in normalized and key != canonical:
                continue
            normalized[canonical] = value

        # Drop fields that are not annotated
        annotated = {k: v for k, v in normalized.items() if k in cls.__annotations__}
        return cls(**annotated)


# --- Behavior rules ----------------------------------------------------------


_BEHAVIOR_RULES: dict[BehaviorIntent, BehaviorResult] = {
    BehaviorIntent.OUT_OF_SCOPE: BehaviorResult(
        intent=BehaviorIntent.OUT_OF_SCOPE.value,
        decision=Decision.NEED_MORE_INFO,
        requires_human=False,
        reasoning="该问题超出当前3C店铺客服范围",
        tool_expectation=None,
    ),
    BehaviorIntent.TOOL_REQUIRED: BehaviorResult(
        intent=BehaviorIntent.TOOL_REQUIRED.value,
        decision=Decision.NEED_MORE_INFO,
        requires_human=False,
        reasoning="需要工具调用查询实时订单/物流信息",
        tool_expectation="query_logistics_or_order",
    ),
    BehaviorIntent.PRIVACY: BehaviorResult(
        intent=BehaviorIntent.PRIVACY.value,
        decision=Decision.ESCALATE,
        requires_human=True,
        reasoning="用户提供了PII/敏感信息，需要安全处理转人工",
        tool_expectation=None,
    ),
    BehaviorIntent.INJECTION: BehaviorResult(
        intent=BehaviorIntent.INJECTION.value,
        decision=Decision.REJECT,
        requires_human=False,
        reasoning="检测到提示注入尝试，已拒答",
        tool_expectation=None,
    ),
    BehaviorIntent.ESCALATE: BehaviorResult(
        intent=BehaviorIntent.ESCALATE.value,
        decision=Decision.ESCALATE,
        requires_human=True,
        reasoning="用户明确要求升级或人工处理",
        tool_expectation=None,
    ),
}


class PolicyEngine:
    """Deterministic policy matching engine.

    The engine maintains per-policy scoring across the full candidate pool and
    only filters out policies whose conditions actively conflict with the
    caller-provided context. A "missing slot" is only meaningful relative to a
    specific policy, so missing_slots is computed against the best partial
    candidate rather than as a global fallback.
    """

    def __init__(self, policies: list[dict[str, Any]]):
        self.policies = policies
        self._validate_policies()
        self._build_index()

    # ----- Indexing ---------------------------------------------------------

    def _validate_policies(self) -> None:
        """Ensure policy IDs and decision enums are well-formed."""
        seen: set = set()
        for p in self.policies:
            pid = p.get("policy_id")
            if not pid:
                raise ValueError("policy without policy_id")
            if pid in seen:
                raise ValueError(f"duplicate policy_id: {pid}")
            seen.add(pid)
            for d in p.get("decisions", []):
                decision = d.get("decision")
                if decision not in Decision._value2member_map_:
                    raise ValueError(
                        f"policy {pid} has unknown decision: {decision}"
                    )

    def _build_index(self) -> None:
        """Index policies by category for faster lookup."""
        self._by_category: dict[str, list[dict[str, Any]]] = {}
        for policy in self.policies:
            category = policy.get("category", "unknown")
            self._by_category.setdefault(category, []).append(policy)

    # ----- Public API -------------------------------------------------------

    def list_categories(self) -> list[str]:
        """Return all known policy categories."""
        return list(self._by_category.keys())

    def match(
        self,
        context: dict[str, Any],
        category_hint: str | None = None,
        accept_aliases: bool = True,
    ) -> PolicyMatch:
        """Match a structured context against policies.

        Args:
            context: Structured slot map (raw dict, may include aliases if
                ``accept_aliases`` is True).
            category_hint: Restrict candidates to a specific category. When
                omitted, all policies are evaluated.
            accept_aliases: Passed through to ``SlotSchema.from_dict``.

        Returns:
            A ``PolicyMatch``. ``policy_id`` is empty when no policy fully
            matches; ``ambiguity`` is set when several candidates tie.
        """
        try:
            slots = SlotSchema.from_dict(context, accept_aliases=accept_aliases)
        except ValueError as e:
            return PolicyMatch(
                policy_id="",
                decision=Decision.ESCALATE,
                reasoning=f"Invalid context: {e}",
                requires_human=True,
                escalation_reasons=["unrecognized_slot"],
            )

        slot_dict = slots.to_dict()

        candidates = self.policies
        if category_hint is not None:
            candidates = self._by_category.get(category_hint, [])
            if not candidates:
                return PolicyMatch(
                    policy_id="",
                    decision=Decision.ESCALATE,
                    reasoning=f"unknown category: {category_hint}",
                    requires_human=True,
                )

        evaluations: list[tuple[dict, dict, set, list[str], list[str]]] = []
        for policy in candidates:
            evaluation = self._evaluate_policy(policy, slot_dict)
            evaluations.append(evaluation)

        # A "matchable" policy is one without known conflicts. Conflicts mean
        # the caller's context actively contradicts the policy, so even when
        # nothing is "missing" we should not consider it a full match.
        matchable = [e for e in evaluations if not e["conflicts"]]
        full_matches = [e for e in matchable if e["missing"] == []]

        if len(full_matches) == 1:
            policy = full_matches[0]["policy"]
            decision = policy["decisions"][0]
            return PolicyMatch(
                policy_id=policy["policy_id"],
                decision=Decision(decision["decision"]),
                reasoning=decision.get("reasoning", ""),
                requires_human=decision.get("requires_human", False),
                candidate_policy_ids=[e["policy"]["policy_id"] for e in full_matches],
                escalation_reasons=decision.get("escalation_reasons", []),
            )

        if len(full_matches) > 1:
            # Tied full match -> ambiguous, escalate
            policy_ids = [e["policy"]["policy_id"] for e in full_matches]
            return PolicyMatch(
                policy_id="",
                decision=Decision.AMBIGUOUS,
                reasoning=f"Multiple full matches: {policy_ids}",
                requires_human=True,
                candidate_policy_ids=policy_ids,
                ambiguity=True,
                escalation_reasons=["ambiguous_full_match"],
            )

        # No full matches. Look for the strongest partial match among
        # matchable policies (no active conflicts).
        any_partial = matchable
        any_partial.sort(
            key=lambda e: (
                -len(e["matched_fields"]),
                len(e["missing"]),
                -e["priority"],
                e["policy"]["policy_id"],  # tie-break with policy id
            )
        )
        if any_partial and any_partial[0]["matched_fields"]:
            best = any_partial[0]
            return PolicyMatch(
                policy_id="",
                decision=Decision.NEED_MORE_INFO,
                reasoning="no policy fully matched",
                requires_human=False,
                missing_slots=list(best["missing"]),
                candidate_policy_ids=[],
                escalation_reasons=[],
            )

        # Nothing matched at all
        return PolicyMatch(
            policy_id="",
            decision=Decision.ESCALATE,
            reasoning="no compatible policy",
            requires_human=True,
            conflicted_policies=self._collect_conflicts(evaluations),
        )

    def match_behavior(self, intent: str) -> BehaviorResult:
        """Resolve a non-policy intent via explicit behavior rules.

        Unknown behavior intents fall back to ``out_of_scope``.
        """
        key = _BEHAVIOR_INTENT_ALIASES.get(intent, BehaviorIntent.OUT_OF_SCOPE)
        return _BEHAVIOR_RULES[key]

    # ----- Internals --------------------------------------------------------

    @staticmethod
    def _evaluate_policy(
        policy: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a policy against a context dict.

        Returns a dict with:
            policy: original policy
            matched_fields: set of condition field names whose comparison succeeded
            matched: count of matched fields
            missing: list of fields that were absent
            conflicts: list of fields whose comparison failed (present but mismatched)
            priority: -len(missing) so policies with fewer missing slots break ties
        """
        conditions = policy.get("conditions", [])
        matched_fields: set = set()
        missing: list[str] = []
        conflicts: list[str] = []

        for cond in conditions:
            field = cond.get("field")
            operator = cond.get("operator")
            expected = cond.get("value")

            actual = context.get(field)
            if actual is None:
                missing.append(field)
                continue
            if PolicyEngine._evaluate_condition(actual, operator, expected):
                matched_fields.add(field)
            else:
                conflicts.append(field)

        priority = policy.get("priority", 0)
        return {
            "policy": policy,
            "matched_fields": matched_fields,
            "matched": len(matched_fields),
            "missing": missing,
            "conflicts": conflicts,
            "priority": priority,
        }

    @staticmethod
    def _evaluate_condition(actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate a single condition."""
        if operator == "eq":
            return actual == expected
        if operator == "ne":
            return actual != expected
        if operator == "gt":
            return actual > expected
        if operator == "lt":
            return actual < expected
        if operator == "gte":
            return actual >= expected
        if operator == "lte":
            return actual <= expected
        if operator == "in":
            return actual in expected
        if operator == "not_in":
            return actual not in expected
        return False

    @staticmethod
    def _collect_conflicts(evaluations: list[dict]) -> list[str]:
        conflicted: list[str] = []
        seen: set = set()
        for e in evaluations:
            if e["conflicts"]:
                pid = e["policy"]["policy_id"]
                if pid not in seen:
                    conflicted.append(pid)
                    seen.add(pid)
        return conflicted


# Module-level validators ---------------------------------------------------


def validate_policy_uniqueness(policies: list[dict]) -> list[str]:
    """Return list of human-readable errors for duplicate policy IDs."""
    errors: list[str] = []
    seen: set = set()
    for policy in policies:
        pid = policy.get("policy_id")
        if not pid:
            errors.append("policy without policy_id")
            continue
        if pid in seen:
            errors.append(f"duplicate policy_id: {pid}")
        seen.add(pid)
    return errors
