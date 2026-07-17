"""
Intent Classifier and Slot Schema for E-commerce Customer Service

Defines the intent taxonomy and slot types for the customer service domain.
"""

from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    """Customer service intent categories."""

    # Logistics
    LOGISTICS_QUERY = "logistics_query"  # 物流查询
    LOGISTICS_EXCEPTION = "logistics_exception"  # 物流异常
    DELIVERY_DELAY = "delivery_delay"  # 发货/配送延迟

    # Returns
    RETURN_QUERY = "return_query"  # 退货咨询
    RETURN_APPLICATION = "return_application"  # 退货申请
    RETURN_STATUS = "return_status"  # 退货进度

    # Exchanges
    EXCHANGE_QUERY = "exchange_query"  # 换货咨询
    EXCHANGE_APPLICATION = "exchange_application"  # 换货申请

    # Specifications
    PRODUCT_SPEC = "product_spec"  # 商品规格
    COMPATIBILITY = "compatibility"  # 兼容性
    STOCK_QUERY = "stock_query"  # 库存查询

    # Coupons
    COUPON_QUERY = "coupon_query"  # 优惠券咨询
    COUPON_ISSUE = "coupon_issue"  # 优惠券问题
    PRICE_PROTECTION = "price_protection"  # 价保申请
    DISCOUNT_QUERY = "discount_query"  # 优惠咨询

    # Complaints
    COMPLAINT = "complaint"  # 投诉
    REFUND_DISPUTE = "refund_dispute"  # 退款争议
    ESCALATE = "escalate"  # 要求转人工

    # Others
    GREETING = "greeting"  # 问候
    GENERAL_QUERY = "general_query"  # 一般咨询
    UNKNOWN = "unknown"  # 无法识别


class Slot(str, Enum):
    """Slot types for extracting structured information."""

    # Order Information
    ORDER_ID = "order_id"  # 订单号
    PRODUCT_NAME = "product_name"  # 商品名称
    PRODUCT_CATEGORY = "product_category"  # 商品类别
    SKU = "sku"  # 商品SKU

    # Time Related
    ORDER_DATE = "order_date"  # 订单日期
    DELIVERY_DATE = "delivery_date"  # 收货日期
    DAYS_SINCE_DELIVERY = "days_since_delivery"  # 收货天数

    # Product Condition
    PACKAGE_STATUS = "package_status"  # 包装状态 (opened/unopened)
    PRODUCT_CONDITION = "product_condition"  # 商品状况
    USER_DAMAGE = "user_damage"  # 是否人为损坏
    ACCESSORIES_COMPLETE = "accessories_complete"  # 配件是否齐全
    QUALITY_ISSUE = "quality_issue"  # 是否有质量问题
    ISSUE_DESCRIPTION = "issue_description"  # 问题描述

    # Logistics
    TRACKING_NUMBER = "tracking_number"  # 快递单号
    LOGISTICS_STATUS = "logistics_status"  # 物流状态
    EXPECTED_DELIVERY = "expected_delivery"  # 预计送达

    # Exchange/Return
    EXCHANGE_REASON = "exchange_reason"  # 换货原因
    RETURN_REASON = "return_reason"  # 退货原因
    PREFERRED_SIZE = "preferred_size"  # 期望尺码
    PREFERRED_COLOR = "preferred_color"  # 期望颜色

    # Coupon/Price
    COUPON_CODE = "coupon_code"  # 优惠券码
    ORDER_AMOUNT = "order_amount"  # 订单金额
    ORIGINAL_PRICE = "original_price"  # 原价
    CURRENT_PRICE = "current_price"  # 现价

    # Customer Info
    USER_EMOTION = "user_emotion"  # 用户情绪
    CUSTOMER_NAME = "customer_name"  # 客户姓名

    # Context
    SESSION_HISTORY = "session_history"  # 会话历史
    PREVIOUS_INTENT = "previous_intent"  # 上一轮意图


@dataclass
class IntentRule:
    """Rule-based intent classification."""

    intent: Intent
    keywords: list[str]
    exclude_keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)


# Intent classification rules (simplified, can be enhanced with ML)
INTENT_RULES = [
    # Logistics
    IntentRule(
        intent=Intent.LOGISTICS_QUERY,
        keywords=["物流", "快递", "发货", "到哪了", "什么时候到", "派送", "运输"],
        exclude_keywords=["延迟", "异常", "丢了"],
        patterns=["物流查询", "快递查询", "发货时间", "预计送达"],
    ),
    IntentRule(
        intent=Intent.LOGISTICS_EXCEPTION,
        keywords=["丢件", "破损", "丢失", "异常"],
        patterns=["物流异常", "丢件", "快递丢失", "包裹丢失"],
    ),
    IntentRule(
        intent=Intent.DELIVERY_DELAY,
        keywords=["延迟", "晚了", "超时", "一直没到"],
        patterns=["发货延迟", "配送延迟", "延迟发货"],
    ),
    # Returns
    IntentRule(
        intent=Intent.RETURN_QUERY,
        keywords=["退货", "能不能退", "可以退吗", "退掉"],
        patterns=["退货咨询", "退货政策", "怎么退货"],
    ),
    IntentRule(
        intent=Intent.RETURN_APPLICATION,
        keywords=["申请退货", "要退货", "想退货"],
        patterns=["退货申请", "申请退"],
    ),
    # Exchanges
    IntentRule(
        intent=Intent.EXCHANGE_QUERY,
        keywords=["换货", "换一下", "换一个", "换尺码", "换颜色"],
        patterns=["换货咨询", "换货政策", "怎么换货"],
    ),
    # Specifications
    IntentRule(
        intent=Intent.PRODUCT_SPEC,
        keywords=["参数", "规格", "尺寸", "材质", "功能", "介绍"],
        patterns=["商品规格", "产品参数", "详细介绍"],
    ),
    IntentRule(
        intent=Intent.COMPATIBILITY,
        keywords=["兼容", "能用吗", "适配", "配对", "支持"],
        patterns=["兼容", "适配", "能不能用"],
    ),
    # Coupons
    IntentRule(
        intent=Intent.COUPON_QUERY,
        keywords=["优惠券", "优惠码", "能不能用券", "红包"],
        patterns=["优惠券", "优惠码", "红包"],
    ),
    IntentRule(
        intent=Intent.PRICE_PROTECTION,
        keywords=["价保", "保价", "降价", "差价", "退差价"],
        patterns=["价格保护", "价保", "差价退款"],
    ),
    # Complaints
    IntentRule(
        intent=Intent.COMPLAINT,
        keywords=["投诉", "不满", "太差", "态度", "欺骗", "虚假宣传"],
        patterns=["投诉", "举报", "差评"],
    ),
    IntentRule(
        intent=Intent.ESCALATE,
        keywords=["转人工", "人工客服", "人工", "找人工", "主管", "经理"],
        patterns=["转人工", "人工服务", "人工客服"],
    ),
    # Greetings
    IntentRule(
        intent=Intent.GREETING, keywords=["你好", "hi", "hello", "在吗", "在不在"], patterns=[]
    ),
]


class IntentClassifier:
    """Simple rule-based intent classifier."""

    def __init__(self):
        self.rules = INTENT_RULES

    def classify(self, query: str) -> tuple[Intent, float]:
        """Classify user query into intent with confidence."""
        query.lower()

        scores = {}
        for rule in self.rules:
            score = 0.0
            matched_keywords = 0

            # Check keywords
            for keyword in rule.keywords:
                if keyword in query:
                    score += 1.0
                    matched_keywords += 1

            # Check patterns
            for pattern in rule.patterns:
                if pattern in query:
                    score += 2.0

            # Check exclusions
            excluded = False
            for excl in rule.exclude_keywords:
                if excl in query:
                    excluded = True
                    break

            if not excluded and score > 0:
                scores[rule.intent] = score

        if not scores:
            return Intent.UNKNOWN, 0.0

        # Return intent with highest score
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]

        # Normalize confidence (rough approximation)
        confidence = min(1.0, max_score / 3.0)

        return best_intent, confidence

    def get_required_slots(self, intent: Intent) -> list[Slot]:
        """Get required slots for a given intent."""
        required_slots_map = {
            Intent.LOGISTICS_QUERY: [Slot.ORDER_ID, Slot.TRACKING_NUMBER],
            Intent.LOGISTICS_EXCEPTION: [
                Slot.ORDER_ID,
                Slot.TRACKING_NUMBER,
                Slot.ISSUE_DESCRIPTION,
            ],
            Intent.RETURN_QUERY: [Slot.PRODUCT_NAME, Slot.DAYS_SINCE_DELIVERY],
            Intent.RETURN_APPLICATION: [Slot.ORDER_ID, Slot.RETURN_REASON, Slot.PRODUCT_CONDITION],
            Intent.EXCHANGE_QUERY: [Slot.PRODUCT_NAME],
            Intent.EXCHANGE_APPLICATION: [Slot.ORDER_ID, Slot.EXCHANGE_REASON],
            Intent.PRODUCT_SPEC: [Slot.PRODUCT_NAME],
            Intent.COMPATIBILITY: [Slot.PRODUCT_NAME],
            Intent.COUPON_QUERY: [Slot.ORDER_AMOUNT],
            Intent.PRICE_PROTECTION: [Slot.ORDER_ID, Slot.ORDER_AMOUNT],
            Intent.COMPLAINT: [Slot.ISSUE_DESCRIPTION, Slot.USER_EMOTION],
            Intent.GREETING: [],
        }
        return required_slots_map.get(intent, [])


def get_slot_extraction_prompts(intent: Intent) -> dict[str, str]:
    """Get prompts for slot extraction based on intent."""
    prompts = {
        Intent.LOGISTICS_QUERY: """从用户查询中提取物流相关信息：
- 订单号（如有）
- 快递单号（如有）
- 具体物流问题""",
        Intent.RETURN_QUERY: """从用户查询中提取退货相关信息：
- 商品名称或类别
- 收货天数
- 商品状况（是否拆封、是否损坏）
- 配件是否齐全
- 退货原因""",
        Intent.EXCHANGE_QUERY: """从用户查询中提取换货相关信息：
- 商品名称或类别
- 换货原因（尺码、颜色、质量）
- 期望的尺码或颜色""",
        Intent.COMPLAINT: """从用户查询中提取投诉相关信息：
- 投诉类型（服务态度、商品问题、虚假宣传等）
- 具体情况描述
- 用户情绪状态""",
    }
    return prompts.get(intent, "提取查询中的关键信息。")


if __name__ == "__main__":
    classifier = IntentClassifier()

    test_queries = [
        "我的耳机还没到，都三天了",
        "耳机拆了包装还能退吗？",
        "这个蓝牙耳机支持苹果手机吗？",
        "我的订单用不了优惠券怎么回事",
        "你好，我想问一下退货政策",
        "我要投诉你们客服，态度太差了",
        "帮我转人工",
    ]

    print("Intent Classification Test:")
    print("-" * 60)
    for query in test_queries:
        intent, conf = classifier.classify(query)
        required_slots = classifier.get_required_slots(intent)
        print(f"Query: {query}")
        print(f"  Intent: {intent.value} (confidence: {conf:.2f})")
        print(f"  Required slots: {[s.value for s in required_slots]}")
        print()
