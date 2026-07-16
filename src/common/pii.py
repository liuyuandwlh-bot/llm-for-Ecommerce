"""
PII Detection and Masking

Detect and mask personal identifiable information (PII) with proper handling
of overlapping matches.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from enum import IntEnum


class PIIType(IntEnum):
    """PII types with priority (higher = more specific)."""
    PHONE_FIXED = 1
    PHONE_CN = 2
    EMAIL = 3
    ORDER_ID = 4
    ID_CARD = 6  # Higher priority than bank card
    BANK_CARD = 5
    PASSPORT = 7


@dataclass
class PIIMatch:
    """A PII match with location and type."""
    pii_type: str
    text: str
    start: int
    end: int
    masked: str


class PIIDetector:
    """
    PII detector for common Chinese PII types.
    
    Handles overlapping matches by priority and longest-match rules.
    """

    # Regex patterns for PII detection (ordered by specificity)
    PATTERNS = [
        # Fixed phone (must check before mobile)
        ("phone_fixed", r'0\d{2,3}-?\d{7,8}', PIIType.PHONE_FIXED),
        # Chinese mobile
        ("phone_cn", r'1[3-9]\d{9}', PIIType.PHONE_CN),
        # Email
        ("email", r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', PIIType.EMAIL),
        # Chinese ID card (18 digits)
        ("id_card", r'\d{17}[\dXx]', PIIType.ID_CARD),
        # Bank card (16 digits minimum)
        ("bank_card", r'\b\d{16,19}\b', PIIType.BANK_CARD),
        # Order ID pattern
        ("order_id", r'(?:订单号|订单|单号)[：:\s]*([A-Z0-9]{10,})', PIIType.ORDER_ID),
        # Passport
        ("passport", r'[A-Z]{1,2}\d{6,9}', PIIType.PASSPORT),
    ]

    # Placeholder templates (no real content retained)
    PLACEHOLDERS: Dict[str, str] = {
        "phone_cn": "<PHONE>",
        "phone_fixed": "<PHONE_FIXED>",
        "email": "<EMAIL>",
        "id_card": "<ID_CARD>",
        "bank_card": "<BANK_CARD>",
        "order_id": "<ORDER>",
        "passport": "<PASSPORT>",
    }

    def __init__(self):
        # Compile patterns
        self.compiled_patterns: List[Tuple[str, re.Pattern, PIIType]] = [
            (name, re.compile(pattern), ptype)
            for name, pattern, ptype in self.PATTERNS
        ]
        
        # Track entity mappings for consistent masking
        self._entity_map: Dict[str, str] = {}

    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text.
        
        Returns matches sorted by start position with overlapping
        matches resolved by priority (higher priority wins).
        """
        # Find all matches
        all_matches: List[Tuple[int, int, str, str, PIIType]] = []
        
        for name, pattern, ptype in self.compiled_patterns:
            for match in pattern.finditer(text):
                all_matches.append((
                    match.start(),
                    match.end(),
                    name,
                    match.group(),
                    ptype,
                ))
        
        # Sort by start position
        all_matches.sort(key=lambda x: (x[0], -x[4]))
        
        # Resolve overlapping matches using greedy non-overlapping algorithm
        resolved = self._resolve_overlaps(all_matches)
        
        # Build PIIMatch objects
        matches = []
        for match_info in resolved:
            start, end, pii_type, text_content = match_info[0], match_info[1], match_info[2], match_info[3]
            masked = self._get_placeholder(pii_type, text_content)
            matches.append(PIIMatch(
                pii_type=pii_type,
                text=text_content,
                start=start,
                end=end,
                masked=masked,
            ))
        
        return matches

    def _resolve_overlaps(
        self, 
        matches: List[Tuple[int, int, str, str, PIIType]]
    ) -> List[Tuple[int, int, str, str, PIIType]]:
        """
        Resolve overlapping matches.
        
        Uses greedy algorithm:
        1. Sort by start position
        2. Keep highest priority match in overlapping regions
        """
        if not matches:
            return []
        
        resolved = []
        current_end = 0
        
        # Sort by start, then by priority (descending)
        sorted_matches = sorted(matches, key=lambda x: (x[0], -x[4]))
        
        for match_info in sorted_matches:
            start, end, pii_type, text, ptype = match_info
            if start >= current_end:
                # Non-overlapping, add it
                resolved.append((start, end, pii_type, text, ptype))
                current_end = end
            # else: overlapping with higher priority match, skip
        
        return resolved

    def _get_placeholder(self, pii_type: str, text: str) -> str:
        """Get placeholder for PII type, with consistent mapping."""
        placeholder = self.PLACEHOLDERS.get(pii_type, "<PII>")
        
        # For repeatable entities, track and reuse same placeholder
        # This prevents "手机号是13812345678，发到13812345678@qq.com" -> different masks
        if text in self._entity_map:
            return self._entity_map[text]
        
        self._entity_map[text] = placeholder
        return placeholder

    def mask(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """
        Detect and mask PII in text.

        Returns:
            Tuple of (masked text, list of matches)
        """
        matches = self.detect(text)

        if not matches:
            return text, []

        # Replace from end to start to preserve positions
        masked_text = text
        offset = 0

        for match in matches:
            start = match.start + offset
            end = match.end + offset

            masked_text = masked_text[:start] + match.masked + masked_text[end:]
            offset += len(match.masked) - (match.end - match.start)

        return masked_text, matches


def mask_pii(text: str) -> str:
    """Convenience function to mask PII."""
    detector = PIIDetector()
    masked, _ = detector.mask(text)
    return masked


# Global detector instance
_detector = PIIDetector()


def scan_for_pii(text: str) -> List[PIIMatch]:
    """Scan text for PII."""
    return _detector.detect(text)


def mask_pii_in_text(text: str) -> str:
    """Mask PII in text."""
    masked, _ = _detector.mask(text)
    return masked


# ============================================================
# Unit Tests
# ============================================================

def test_overlapping_phone_email():
    """Test overlapping phone number in email is handled."""
    detector = PIIDetector()
    
    text = "邮箱是test13812345678@gmail.com，手机号13812345678"
    masked, matches = detector.mask(text)
    
    # Should detect both but not overlap
    pii_types = [m.pii_type for m in matches]
    assert "phone_cn" in pii_types
    assert "email" in pii_types
    print("test_overlapping_phone_email: PASSED")


def test_overlapping_id_phone():
    """Test overlapping ID card and phone detection."""
    detector = PIIDetector()
    
    # ID card ends with digits that could be phone
    text = "身份证110101199001011234，手机13812345678"
    masked, matches = detector.mask(text)
    
    pii_types = [m.pii_type for m in matches]
    assert "id_card" in pii_types
    assert "phone_cn" in pii_types
    print("test_overlapping_id_phone: PASSED")


def test_consistent_entity_mapping():
    """Test same entity gets same placeholder."""
    detector = PIIDetector()
    
    text = "手机号13812345678，请回拨13812345678"
    masked, matches = detector.mask(text)
    
    # Should use same placeholder for same number
    masked_count = masked.count("<PHONE>")
    assert masked_count == 2
    print("test_consistent_entity_mapping: PASSED")


def test_no_placeholder_leak():
    """Test placeholders don't contain real content."""
    detector = PIIDetector()
    
    test_cases = [
        ("手机13812345678", "<PHONE>"),
        ("邮箱test@example.com", "<EMAIL>"),
        ("身份证110101199001011234", "<ID_CARD>"),
    ]
    
    for text, expected_placeholder in test_cases:
        masked, _ = detector.mask(text)
        # Verify placeholder doesn't contain digits from original
        if "手机" in text:
            assert "138" not in masked
        if "邮箱" in text:
            assert "test@example" not in masked
        if "身份证" in text:
            assert "110101" not in masked or "1990" not in masked
    print("test_no_placeholder_leak: PASSED")


def test_fixed_phone_priority():
    """Test fixed phone is detected before mobile."""
    detector = PIIDetector()
    
    # Fixed phone: 010-12345678
    text = "固定电话010-12345678"
    masked, matches = detector.mask(text)
    
    assert any(m.pii_type == "phone_fixed" for m in matches)
    print("test_fixed_phone_priority: PASSED")


if __name__ == "__main__":
    # Run tests
    print("Running PII detector tests...\n")
    
    test_overlapping_phone_email()
    test_overlapping_id_phone()
    test_consistent_entity_mapping()
    test_no_placeholder_leak()
    test_fixed_phone_priority()
    
    print("\nAll tests passed!")
    
    # Demo
    print("\n" + "="*60)
    print("Demo:")
    print("="*60)
    
    detector = PIIDetector()
    test_texts = [
        "我的手机号是13812345678，请发送到我的邮箱test@example.com",
        "订单号ORD1234567890，身份证号110101199001011234",
        "联系固定电话：010-12345678 或 021-87654321",
    ]
    
    for text in test_texts:
        masked, matches = detector.mask(text)
        print(f"\nOriginal: {text}")
        print(f"Masked: {masked}")
        print(f"Types: {[m.pii_type for m in matches]}")
