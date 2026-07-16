"""Unit tests for the E-commerce dataset module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.ecommerce.dataset import (
    SOPBuilder,
    PolicyEngine,
    CanonicalCaseGenerator,
    ConversationGenerator,
    PipelineConfig,
    Decision,
    SlotSchema,
    validate_canonical_cases,
    run_pipeline,
)


class TestSOPBuilder:
    def test_build_policies(self):
        builder = SOPBuilder()
        policies = builder.build_fictional_store_sops()

        # Stable unique IDs
        ids = [p.policy_id for p in policies]
        assert len(ids) == len(set(ids))

        # Every policy has the required metadata.
        for p in policies:
            assert p.policy_id
            assert p.category
            assert p.version
            assert p.effective_from

    def test_policy_save_load(self, tmp_path):
        builder = SOPBuilder()
        builder.build_fictional_store_sops()
        path = tmp_path / "policies.json"
        builder.save_policies(str(path))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert len(loaded) == len(builder.policies)
        assert all(p["policy_id"] for p in loaded)


class TestPolicyEngine:
    def test_match_full(self):
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
                ],
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
        assert not result.missing_slots
        # ``result.decision`` is a Decision enum whose value identifies the
        # underlying SOP decision string.
        assert result.decision.value == "full_refund"

    def test_match_missing_slot(self):
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
                ],
            }
        ]
        engine = PolicyEngine(policies)
        result = engine.match({"days_since_delivery": 3})
        assert result.missing_slots and "package_status" in result.missing_slots


class TestCanonicalCases:
    def test_generate_all_cases(self):
        gen = CanonicalCaseGenerator()
        cases = gen.generate_all()
        # Contract: at least 20 cases including policy-backed + behavior.
        assert len(cases) >= 20
        case_ids = [c.case_id for c in cases]
        assert len(case_ids) == len(set(case_ids))

    def test_case_save_load(self, tmp_path):
        gen = CanonicalCaseGenerator()
        cases = gen.generate_all()
        path = tmp_path / "cases.jsonl"
        gen.save_cases(cases, str(path))
        loaded = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(loaded) == len(cases)


class TestConversationGenerator:
    def test_generates_conversation_per_strategy(self, tmp_path):
        builder = SOPBuilder()
        policies = [p.to_dict() for p in builder.build_fictional_store_sops()]
        gen = CanonicalCaseGenerator()
        cases = [c.to_dict() for c in gen.generate_all()]

        out = tmp_path / "convs.jsonl"
        cg = ConversationGenerator(
            policies=policies,
            seed=42,
            mode="fixture",
            source_id="owned_sop_v1",
        )
        cg.run(str(out), cases=cases)

        loaded = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # At least one conversation per case (some cases produce more
        # through their declared rewrite strategies).
        assert len(loaded) >= len(cases)
        for c in loaded:
            roles = [m["role"] for m in c["messages"]]
            assert roles[0] == "system"
            assert roles[-1] == "assistant"
            for a, b in zip(roles, roles[1:]):
                assert a != b

    def test_seed_reproducibility(self, tmp_path):
        builder = SOPBuilder()
        policies = [p.to_dict() for p in builder.build_fictional_store_sops()]
        gen = CanonicalCaseGenerator()
        cases = [c.to_dict() for c in gen.generate_all()]

        out1 = tmp_path / "c1.jsonl"
        out2 = tmp_path / "c2.jsonl"
        cg1 = ConversationGenerator(policies=policies, seed=11,
                                    mode="fixture", source_id="owned_sop_v1")
        cg2 = ConversationGenerator(policies=policies, seed=11,
                                    mode="fixture", source_id="owned_sop_v1")
        cg1.run(str(out1), cases=cases)
        cg2.run(str(out2), cases=cases)
        assert out1.read_bytes() == out2.read_bytes()


class TestDataPipeline:
    def test_pipeline_process_runs(self, tmp_path):
        builder = SOPBuilder()
        policies = [p.to_dict() for p in builder.build_fictional_store_sops()]
        gen = CanonicalCaseGenerator()
        cases = [c.to_dict() for c in gen.generate_all()]

        policies_path = tmp_path / "policies.json"
        policies_path.write_text(json.dumps(policies, ensure_ascii=False))

        cases_path = tmp_path / "cases.jsonl"
        gen.save_cases(cases, str(cases_path))

        conv_path = tmp_path / "conv.jsonl"
        ConversationGenerator(policies=policies, seed=42,
                              mode="fixture", source_id="owned_sop_v1").run(str(conv_path), cases=cases)

        registry = {
            "sources": [
                {
                    "source_id": "owned_sop_v1",
                    "source_name": "Owned SOP Policies v1",
                    "license": "internal",
                    "allowed_train": True,
                    "allowed_evaluate": True,
                    "status": "validated",
                    "checksum_sha256": "deadbeef",
                }
            ]
        }
        registry_path = tmp_path / "registry.json"
        registry_path.write_text(json.dumps(registry, ensure_ascii=False))

        config = PipelineConfig(seed=42, mode="fixture")
        out = tmp_path / "out"
        result = run_pipeline(
            str(conv_path),
            str(policies_path),
            str(registry_path),
            str(out),
            config,
        )
        assert result["exit_ok"] is True
        assert result["funnel"]["stage_counts"]["final"] > 0


class TestSlotSchema:
    def test_field_normalization(self):
        slots = SlotSchema.from_dict({"damage_by_user": True, "product_name": "耳机"})
        assert slots.user_damage is True
        assert slots.product_name == "耳机"

    def test_to_dict_roundtrip(self):
        slots = SlotSchema(days_since_delivery=5, package_status="unopened")
        d = slots.to_dict()
        assert d["days_since_delivery"] == 5
        assert d["package_status"] == "unopened"


class TestValidateCanonicalCases:
    def test_validator_passes_for_generated(self):
        builder = SOPBuilder()
        policies = [p.to_dict() for p in builder.build_fictional_store_sops()]
        gen = CanonicalCaseGenerator()
        cases = [c.to_dict() for c in gen.generate_all()]
        severity, errors = validate_canonical_cases(policies, cases)
        assert severity == 0, errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
