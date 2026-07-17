"""Integration tests for the complete Round 2 pipeline."""

import json

import pytest

from src.ecommerce.dataset import (
    CanonicalCaseGenerator,
    PipelineConfig,
    PolicyEngine,
    SOPBuilder,
    run_pipeline,
    validate_canonical_cases,
)
from src.ecommerce.dataset.conversation_generator import (
    ConversationGenerator,
)


class TestPolicyCaseRoundtrip:
    def test_sop_to_cases_to_validate(self, tmp_path):
        builder = SOPBuilder()
        policies = builder.build_fictional_store_sops()
        policies_dict = [p.to_dict() for p in policies]
        policies_path = tmp_path / "policies.json"
        policies_path.write_text(json.dumps(policies_dict, ensure_ascii=False), encoding="utf-8")

        gen = CanonicalCaseGenerator()
        cases = gen.generate_all()
        cases_path = tmp_path / "cases.jsonl"
        gen.save_cases(cases, str(cases_path))

        loaded = [
            json.loads(line)
            for line in cases_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(loaded) == len(cases) > 0

        severity, errors = validate_canonical_cases(policies_dict, loaded)
        assert severity == 0, errors

        _ = PolicyEngine(policies_dict)
        engine_categories = {p["policy_id"] for p in policies_dict}
        referenced = set()
        for case in loaded:
            for pid in case.get("expected_policy_ids") or []:
                if pid:
                    referenced.add(pid)
        assert referenced.issubset(engine_categories), referenced - engine_categories


class TestConversationGeneratorCLI:
    def test_conversation_generator_cli_runs(self, tmp_path):
        builder = SOPBuilder()
        policies = [p.to_dict() for p in builder.build_fictional_store_sops()]
        policies_path = tmp_path / "policies.json"
        policies_path.write_text(json.dumps(policies, ensure_ascii=False), encoding="utf-8")

        gen = CanonicalCaseGenerator()
        cases = [c.to_dict() for c in gen.generate_all()]
        cases_path = tmp_path / "cases.jsonl"
        gen.save_cases(cases, str(cases_path))

        out_path = tmp_path / "convs.jsonl"
        cg = ConversationGenerator(
            policies=policies,
            seed=42,
            mode="fixture",
            source_id="owned_sop_v1",
        )
        cg.run(str(out_path), cases=cases)

        lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) >= 1
        first = json.loads(lines[0])
        assert "messages" in first and first["messages"]
        assert first["messages"][0]["role"] == "system"
        # Subsequent messages must alternate user/assistant and end on assistant.
        roles = [m["role"] for m in first["messages"]]
        assert roles[-1] == "assistant"
        assert roles[0] == "system"
        # No two adjacent same-role messages.
        for a, b in zip(roles, roles[1:], strict=False):
            assert a != b


class TestDataPipelineRoundtrip:
    def test_data_pipeline_process(self, tmp_path):
        builder = SOPBuilder()
        policies = [p.to_dict() for p in builder.build_fictional_store_sops()]
        gen = CanonicalCaseGenerator()
        cases = [c.to_dict() for c in gen.generate_all()]
        cases_path = tmp_path / "cases.jsonl"
        gen.save_cases(cases, str(cases_path))

        out_path = tmp_path / "convs.jsonl"
        ConversationGenerator(
            policies=policies,
            seed=7,
            mode="fixture",
            source_id="owned_sop_v1",
        ).run(str(out_path), cases=cases)

        policy_path = tmp_path / "policies.json"
        policy_path.write_text(json.dumps(policies, ensure_ascii=False), encoding="utf-8")

        registry_path = tmp_path / "registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "sources": [
                        {
                            "source_id": "owned_sop_v1",
                            "source_name": "Owned SOP Policies v1",
                            "license": "internal",
                            "allowed_train": True,
                            "allowed_evaluate": True,
                            "checksum_sha256": "deadbeef",
                            "status": "validated",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        output_dir = tmp_path / "out"
        config = PipelineConfig(seed=42, mode="fixture")
        result = run_pipeline(
            str(out_path),
            str(policy_path),
            str(registry_path),
            str(output_dir),
            config,
        )

        assert result["exit_ok"] is True
        funnel = result["funnel"]
        # All training-eligible samples should reach the output stage.
        assert funnel["stage_counts"]["final"] > 0
        # And there must be no policy inconsistencies.
        assert funnel.get("policy_inconsistent", 0) == 0
        assert funnel.get("invalid_roles", 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
