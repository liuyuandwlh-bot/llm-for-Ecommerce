"""
Policy Engine for E-commerce Customer Service

Deterministic policy matching based on structured context.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


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
    COUPON_APPLICABLE = "coupon_applicable"
    COUPON_NOT_APPLICABLE = "coupon_not_applicable"
    REFUND_DIFFERENCE = "refund_difference"
    SHIP_WITHIN_48H = "ship_within_48h"
    CANCEL_WITH_COMPENSATION = "cancel_with_compensation"
    RESEND_OR_REFUND = "resend_or_refund"
    ACKNOWLEDGE_APOLOGIZE = "acknowledge_apologize"
    ACCEPT_RETURN_REFUND = "accept_return_refund"


@dataclass
class PolicyMatch:
    """Result of policy matching."""
    policy_id: str
    decision: Decision
    reasoning: str
    requires_human: bool
    missing_slots: List[str] = field(default_factory=list)
    escalation_reasons: List[str] = field(default_factory=list)


@dataclass
class SlotSchema:
    """Canonical slot definitions for 3C store."""
    # Order-related
    order_id: Optional[str] = None
    order_date: Optional[str] = None
    delivery_date: Optional[str] = None
    days_since_delivery: Optional[int] = None
    days_since_signed: Optional[int] = None
    
    # Product-related
    product_name: Optional[str] = None
    product_category: Optional[str] = None  # headphones, charger, cable, phone_case
    
    # Package status
    package_status: Optional[str] = None  # unopened, opened
    accessories_complete: Optional[bool] = None
    
    # Condition
    user_damage: Optional[bool] = None  # NOTE: Use this consistently
    # Legacy alias - will be normalized
    damage_by_user: Optional[bool] = None
    
    # Quality
    quality_issue: Optional[bool] = None
    has_proof: Optional[bool] = None
    misleading_confirmed: Optional[bool] = None
    
    # Exchange
    exchange_type: Optional[str] = None  # size, color
    preferred_color: Optional[str] = None
    preferred_size: Optional[str] = None
    
    # Logistics
    logistics_status: Optional[str] = None  # in_transit, delivered, signed, exception
    shipped: Optional[bool] = None
    user_received: Optional[bool] = None
    order_type: Optional[str] = None  # in_stock, preorder
    
    # Coupon/Price
    price_dropped: Optional[bool] = None
    same_product: Optional[bool] = None
    order_amount: Optional[float] = None
    coupon_code: Optional[str] = None
    
    # Complaint
    complaint_type: Optional[str] = None  # service_attitude, false_advertising
    user_emotion: Optional[str] = None  # calm, angry, frustrated
    
    # Compatibility
    user_asked_compatibility: Optional[bool] = None
    target_device: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, normalizing field names."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                # Normalize damage_by_user -> user_damage
                if key == "damage_by_user" and self.user_damage is None:
                    result["user_damage"] = value
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlotSchema":
        """Create from dict, handling legacy field names."""
        # Normalize damage_by_user -> user_damage
        if "damage_by_user" in data and "user_damage" not in data:
            data = dict(data)
            data["user_damage"] = data.pop("damage_by_user")
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class PolicyEngine:
    """
    Deterministic policy matching engine.
    
    Takes structured context as input and returns matching policy decision.
    """
    
    def __init__(self, policies: List[Dict[str, Any]]):
        """
        Initialize with policy list.
        
        Args:
            policies: List of policy dictionaries with conditions and decisions
        """
        self.policies = policies
        self._build_index()
    
    def _build_index(self):
        """Build internal index for efficient matching."""
        # Index by category for faster lookup
        self.policies_by_category: Dict[str, List[Dict]] = {}
        for policy in self.policies:
            cat = policy.get("category", "unknown")
            if cat not in self.policies_by_category:
                self.policies_by_category[cat] = []
            self.policies_by_category[cat].append(policy)
    
    def match(
        self, 
        context: Dict[str, Any],
        category_hint: Optional[str] = None
    ) -> PolicyMatch:
        """
        Match context against policies.
        
        Args:
            context: Structured context with slot values
            category_hint: Optional category to narrow search
            
        Returns:
            PolicyMatch with decision and reasoning
        """
        # Normalize context
        slots = SlotSchema.from_dict(context)
        slot_dict = slots.to_dict()
        
        # Determine category from context if not provided
        if category_hint is None:
            category_hint = self._infer_category(slot_dict)
        
        # Get policies for category
        policies_to_check = self.policies_by_category.get(
            category_hint, 
            self.policies
        )
        
        # Check each policy
        for policy in policies_to_check:
            match_result = self._check_policy(policy, slot_dict)
            if match_result is not None:
                return match_result
        
        # No match found - return need_more_info or escalate
        return self._no_match_response(slot_dict)
    
    def _infer_category(self, context: Dict[str, Any]) -> str:
        """Infer policy category from context."""
        # Check for specific slots that indicate category
        if context.get("days_since_delivery") is not None:
            if context.get("quality_issue"):
                return "return"
            if context.get("exchange_type") is not None:
                return "exchange"
            return "return"
        
        if context.get("logistics_status") is not None or context.get("shipped") is not None:
            return "logistics"
        
        if context.get("price_dropped") is not None or context.get("coupon_code") is not None:
            return "coupon"
        
        if context.get("complaint_type") is not None or context.get("user_emotion") is not None:
            return "complaint"
        
        if context.get("user_asked_compatibility") is not None:
            return "specification"
        
        # Default to return for delivery-related queries
        if context.get("order_id") is not None:
            return "return"
        
        return "return"  # default
    
    def _check_policy(
        self, 
        policy: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Optional[PolicyMatch]:
        """
        Check if context matches a policy's conditions.
        
        Returns PolicyMatch if matched, None otherwise.
        """
        conditions = policy.get("conditions", [])
        
        # Track missing slots
        missing_slots = []
        all_match = True
        
        for cond in conditions:
            field = cond.get("field")
            operator = cond.get("operator")
            expected = cond.get("value")
            
            # Check if slot exists
            actual = context.get(field)
            if actual is None:
                missing_slots.append(field)
                all_match = False
                continue
            
            # Check condition
            if not self._evaluate_condition(actual, operator, expected):
                all_match = False
                break
        
        if not all_match:
            return None
        
        # Check if decision requires human
        decisions = policy.get("decisions", [])
        if not decisions:
            return None
        
        decision = decisions[0]
        
        return PolicyMatch(
            policy_id=policy.get("policy_id", ""),
            decision=Decision(decision.get("decision", "need_more_info")),
            reasoning=decision.get("reasoning", ""),
            requires_human=decision.get("requires_human", False),
            missing_slots=missing_slots,
            escalation_reasons=decision.get("escalation_reasons", []),
        )
    
    def _evaluate_condition(
        self, 
        actual: Any, 
        operator: str, 
        expected: Any
    ) -> bool:
        """Evaluate a single condition."""
        if operator == "eq":
            return actual == expected
        elif operator == "ne":
            return actual != expected
        elif operator == "gt":
            return actual > expected
        elif operator == "lt":
            return actual < expected
        elif operator == "gte":
            return actual >= expected
        elif operator == "lte":
            return actual <= expected
        elif operator == "in":
            return actual in expected
        elif operator == "not_in":
            return actual not in expected
        return False
    
    def _no_match_response(self, context: Dict[str, Any]) -> PolicyMatch:
        """Generate response when no policy matches."""
        # Check if we have minimum required info
        has_order_info = context.get("order_id") is not None
        has_product_info = context.get("product_name") is not None
        
        if not has_order_info or not has_product_info:
            missing = []
            if not has_order_info:
                missing.append("order_id")
            if not has_product_info:
                missing.append("product_name")
            
            return PolicyMatch(
                policy_id="",
                decision=Decision.NEED_MORE_INFO,
                reasoning="需要更多信息才能处理您的请求",
                requires_human=False,
                missing_slots=missing,
            )
        
        # Unclear case - escalate
        return PolicyMatch(
            policy_id="",
            decision=Decision.ESCALATE,
            reasoning="您的问题需要人工客服处理",
            requires_human=True,
            escalation_reasons=["无法匹配现有政策"],
        )


def validate_policy_references(
    policies: List[Dict],
    cases: List[Dict]
) -> List[str]:
    """
    Validate that all case policy references exist in policies.
    
    Returns list of error messages.
    """
    errors = []
    policy_ids = {p["policy_id"] for p in policies}
    
    for case in cases:
        case_id = case.get("case_id", "unknown")
        expected_policies = case.get("expected_policy_ids", [])
        
        for policy_id in expected_policies:
            if policy_id and policy_id not in policy_ids:
                errors.append(
                    f"Case {case_id}: referenced policy {policy_id} not found"
                )
    
    return errors


def validate_policy_uniqueness(policies: List[Dict]) -> List[str]:
    """
    Validate that policy IDs are unique.
    
    Returns list of error messages.
    """
    errors = []
    seen_ids = set()
    
    for policy in policies:
        policy_id = policy.get("policy_id")
        if policy_id in seen_ids:
            errors.append(f"Duplicate policy_id: {policy_id}")
        seen_ids.add(policy_id)
    
    return errors
