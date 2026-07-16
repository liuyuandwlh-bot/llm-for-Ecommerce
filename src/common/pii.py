"""
PII Detection and Masking

Detect and mask personal identifiable information.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


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

    Types detected:
    - Phone numbers
    - Email addresses
    - ID numbers
    - Bank card numbers
    - Order numbers (partial)
    """

    # Regex patterns for PII detection
    PATTERNS = {
        "phone_cn": r'1[3-9]\d{9}',  # Chinese mobile
        "phone_fixed": r'0\d{2,3}-?\d{7,8}',  # Fixed phone
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "id_card": r'\d{17}[\dXx]',  # Chinese ID card
        "bank_card": r'\d{16,19}',  # Bank card
        "order_id": r'(?:订单号|订单)[：:\s]*([A-Z0-9]{10,})',  # Order IDs
        "passport": r'[A-Z]\d{8,9}',  # Passport
    }

    def __init__(self):
        self.compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.PATTERNS.items()
        }

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect all PII in text."""
        matches = []

        for pii_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                masked = self._mask(pii_type, match.group())
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    masked=masked,
                ))

        # Sort by position
        matches.sort(key=lambda x: x.start)
        return matches

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

    def _mask(self, pii_type: str, text: str) -> str:
        """Generate masked version of PII."""
        masks = {
            "phone_cn": f"<PHONE_{len([m for m in text if m.isdigit()])}>",
            "phone_fixed": "<PHONE_FIXED>",
            "email": f"<EMAIL_{text.split('@')[0][:2]}>",
            "id_card": f"<ID_{text[:6]}****{text[-4:]}>",
            "bank_card": f"<BANK_CARD_{len(text)}>",
            "order_id": f"<ORDER_{len(text)}>",
            "passport": "<PASSPORT>",
        }
        return masks.get(pii_type, "<PII>")


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


if __name__ == "__main__":
    # Test
    test_texts = [
        "我的手机号是13812345678，请发送到我的邮箱test@example.com",
        "订单号ORD1234567890，身份证号110101199001011234",
        "联系地址：北京市朝阳区xxx街1号",
    ]

    detector = PIIDetector()
    for text in test_texts:
        masked, matches = detector.mask(text)
        print(f"Original: {text}")
        print(f"Masked: {masked}")
        print(f"Matches: {[m.pii_type for m in matches]}")
        print()
