"""
Tests for Common Utilities

Tests PII detection, schemas, and logging.
"""

import pytest

from src.common.pii import PIIDetector, mask_pii_in_text, scan_for_pii
from src.common.schemas import (
    CustomerServiceRequest,
    CustomerServiceResponse,
    HealthResponse,
)


class TestPIIDetector:
    """Test PII detection and masking."""

    def test_phone_detection(self):
        """Test Chinese phone number detection."""
        detector = PIIDetector()
        
        text = "我的手机号是13812345678"
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == "phone_cn"

    def test_email_detection(self):
        """Test email detection."""
        detector = PIIDetector()
        
        text = "邮箱是test@example.com"
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == "email"

    def test_id_card_detection(self):
        """Test Chinese ID card detection."""
        detector = PIIDetector()
        
        text = "身份证号110101199001011234"
        matches = detector.detect(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == "id_card"

    def test_overlapping_matches(self):
        """Test overlapping PII matches are resolved."""
        detector = PIIDetector()
        
        # Email contains digits that could match phone
        text = "邮箱test13812345678@example.com，手机13812345678"
        matches = detector.detect(text)
        
        # Should detect both but not overlap incorrectly
        pii_types = set(m.pii_type for m in matches)
        assert "phone_cn" in pii_types
        assert "email" in pii_types

    def test_mask_preserves_non_pii(self):
        """Test that non-PII text is preserved."""
        detector = PIIDetector()
        
        text = "耳机收到三天了，还没拆封，能退货吗？"
        masked, matches = detector.mask(text)
        
        assert masked == text
        assert len(matches) == 0

    def test_consistent_entity_masking(self):
        """Test same entity gets same placeholder."""
        detector = PIIDetector()
        
        text = "手机号13812345678，请回拨13812345678"
        masked, _ = detector.mask(text)
        
        # Should use same placeholder for same number
        assert masked.count("<PHONE>") == 2

    def test_no_content_leak_in_placeholder(self):
        """Test placeholders don't contain real content."""
        detector = PIIDetector()
        
        text = "手机13812345678"
        masked, _ = detector.mask(text)
        
        # Placeholder should not contain the actual number
        assert "13812345678" not in masked
        assert "<PHONE>" in masked


class TestSchemas:
    """Test Pydantic schemas."""

    def test_customer_service_request(self):
        """Test customer service request schema."""
        req = CustomerServiceRequest(
            query="耳机能退吗？",
            domain="ecommerce",
        )
        
        assert req.query == "耳机能退吗？"
        assert req.domain == "ecommerce"

    def test_health_response(self):
        """Test health response schema."""
        resp = HealthResponse(
            status="healthy",
            version="0.2.0",
            uptime_seconds=10.5,
            models_loaded=["base_model"],
        )
        
        assert resp.status == "healthy"
        assert resp.models_loaded == ["base_model"]


class TestMaskPIIFunction:
    """Test convenience functions."""

    def test_mask_pii_function(self):
        """Test mask_pii convenience function."""
        text = "手机号13812345678"
        masked = mask_pii_in_text(text)
        
        assert masked == "手机号<PHONE>"

    def test_scan_pii_function(self):
        """Test scan_for_pii convenience function."""
        text = "邮箱test@example.com"
        matches = scan_for_pii(text)
        
        assert len(matches) == 1
        assert matches[0].pii_type == "email"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
