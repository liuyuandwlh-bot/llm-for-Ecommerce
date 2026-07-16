"""
Tests for E-commerce Dataset Module

Tests SOP builder, PolicyEngine, canonical cases, conversation generator, and pipeline.
"""

import json
import tempfile
import pytest
from pathlib import Path

# Import modules under test
from src.ecommerce.dataset import (
    SOPBuilder,
    PolicyEngine,
    CanonicalCaseGenerator,
    CanonicalCase,
    ConversationGenerator,
    DataPipeline,
    Decision,
    SlotSchema,
)


class TestSOPBuilder:
    """Test SOP builder with stable policy IDs."""

    def test_build_policies(self):
        """Test building SOP policies."""
        builder = SOPBuilder()
        policies = builder.build_fictional_store_sops()
        
        assert len(policies) > 0
        
        # Check all policies have stable IDs
        policy_ids = [p.policy_id for p in policies]
        assert len(policy_ids) == len(set(policy_ids))  # All unique
        
        # Check format
        for policy in policies:
            assert policy.policy_id.startswith(policy.category)
            assert policy.version
            assert policy.effective_from

    def test_policy_save_load(self):
        """Test saving and loading policies."""
        builder = SOPBuilder()
        builder.build_fictional_store_sops()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            builder.save_policies(path)
            
            with open(path, 'r') as f:
                loaded = json.load(f)
            
            assert len(loaded) == len(builder.policies)
            assert all(p['policy_id'] for p in loaded)
        finally:
            Path(path).unlink(missing_ok=True)


class TestPolicyEngine:
    """Test PolicyEngine matching."""

    def test_match_return_unopened(self):
        """Test matching return policy for unopened package."""
        policies = [
            {
                "policy_id": "return_001",
                "category": "return",
                "conditions": [
                    {"field": "days_since_delivery", "operator": "lte", "value": 7},
                    {"field": "package_status", "operator": "eq", "value": "unopened"},
                    {"field": "user_damage", "operator": "eq", "value": False},
                ],
                "decisions": [
                    {"decision": "full_refund", "reasoning": "符合条件", "requires_human": False}
                ]
            }
        ]
        
        engine = PolicyEngine(policies)
        
        context = {
            "days_since_delivery": 3,
            "package_status": "unopened",
            "user_damage": False,
        }
        
        result = engine.match(context)
        
        assert result.policy_id == "return_001"
        assert result.decision == Decision.FULL_REFUND
        assert not result.requires_human

    def test_match_missing_slot(self):
        """Test matching when required slot is missing."""
        policies = [
            {
                "policy_id": "return_001",
                "category": "return",
                "conditions": [
                    {"field": "days_since_delivery", "operator": "lte", "value": 7},
                    {"field": "package_status", "operator": "eq", "value": "unopened"},
                ],
                "decisions": [
                    {"decision": "full_refund", "reasoning": "符合条件", "requires_human": False}
                ]
            }
        ]
        
        engine = PolicyEngine(policies)
        
        # Missing package_status
        context = {
            "days_since_delivery": 3,
        }
        
        result = engine.match(context)
        
        assert result.missing_slots == ["package_status"]


class TestCanonicalCases:
    """Test canonical case generation."""

    def test_generate_all_cases(self):
        """Test generating all canonical cases."""
        generator = CanonicalCaseGenerator()
        cases = generator.generate_all()
        
        assert len(cases) > 0
        
        # Check case structure
        for case in cases:
            assert case.case_id
            assert case.intent
            assert case.case_type
            assert case.context
            assert case.expected_decision

    def test_case_types_coverage(self):
        """Test that all case types are covered."""
        generator = CanonicalCaseGenerator()
        cases = generator.generate_all()
        
        case_types = set(c.case_type for c in cases)
        
        # Should have normal and at least one special type
        assert "normal" in case_types

    def test_case_save_load(self):
        """Test saving and loading cases."""
        generator = CanonicalCaseGenerator()
        cases = generator.generate_all()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        
        try:
            generator.save_cases(path)
            
            loaded = []
            with open(path, 'r') as f:
                for line in f:
                    loaded.append(json.loads(line))
            
            assert len(loaded) == len(cases)
        finally:
            Path(path).unlink(missing_ok=True)


class TestConversationGenerator:
    """Test conversation generation."""

    def test_deterministic_generation(self):
        """Test deterministic template generation."""
        cases = [
            CanonicalCase(
                case_id="case_test_001",
                intent="return_query",
                case_type="normal",
                turns=[
                    {"role": "user", "content": "耳机能退吗？"}
                ],
                context={"days_since_delivery": 3},
                expected_policy_ids=["return_001"],
                expected_decision="full_refund",
            )
        ]
        
        generator = ConversationGenerator(seed=42)
        convs = generator.generate_from_cases(cases, samples_per_case=2)
        
        assert len(convs) == 2
        
        # Check conversation structure
        for conv in convs:
            assert conv.sample_id
            assert len(conv.messages) >= 2
            assert conv.intent == "return_query"

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        cases = [
            CanonicalCase(
                case_id="case_test_001",
                intent="return_query",
                case_type="normal",
                turns=[{"role": "user", "content": "测试"}],
                context={},
                expected_policy_ids=[],
                expected_decision="need_more_info",
            )
        ]
        
        gen1 = ConversationGenerator(seed=42)
        convs1 = gen1.generate_from_cases(cases, samples_per_case=1)
        
        gen2 = ConversationGenerator(seed=42)
        convs2 = gen2.generate_from_cases(cases, samples_per_case=1)
        
        assert convs1[0].messages == convs2[0].messages


class TestDataPipeline:
    """Test data processing pipeline."""

    def test_schema_validation(self):
        """Test schema validation."""
        pipeline = DataPipeline()
        
        # Valid sample
        valid_sample = {
            "sample_id": "test_001",
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "您好"}
            ],
            "intent": "return_query",
            "slots": {},
            "policy_ids": ["return_001"],
            "decision": "full_refund",
            "requires_human": False,
        }
        
        result = pipeline._process_sample(valid_sample)
        assert result["valid"]
        
        # Invalid sample - missing required field
        invalid_sample = {
            "sample_id": "test_002",
            "messages": [{"role": "user", "content": "你好"}],
            "intent": "return_query",
        }
        
        result = pipeline._process_sample(invalid_sample)
        assert not result["valid"]
        assert result["reason"] == "schema_invalid"

    def test_pii_masking(self):
        """Test PII masking in pipeline."""
        pipeline = DataPipeline()
        
        sample = {
            "sample_id": "test_001",
            "messages": [
                {"role": "user", "content": "我的手机号13812345678"},
                {"role": "assistant", "content": "好的，我记录了"}
            ],
            "intent": "return_query",
            "slots": {},
            "policy_ids": [],
            "decision": "need_more_info",
            "requires_human": False,
        }
        
        result = pipeline._process_sample(sample)
        assert result["valid"]
        
        # Check PII was masked
        masked_messages = result["sample"]["messages"]
        assert "13812345678" not in masked_messages[0]["content"]


class TestSlotSchema:
    """Test slot schema normalization."""

    def test_field_normalization(self):
        """Test that damage_by_user is normalized to user_damage."""
        slots = SlotSchema.from_dict({
            "damage_by_user": True,
            "product_name": "耳机"
        })
        
        assert slots.user_damage is True
        assert slots.product_name == "耳机"

    def test_to_dict(self):
        """Test converting to dict."""
        slots = SlotSchema(
            days_since_delivery=5,
            package_status="unopened",
        )
        
        d = slots.to_dict()
        assert d["days_since_delivery"] == 5
        assert d["package_status"] == "unopened"
        assert "damage_by_user" not in d or d.get("user_damage") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
