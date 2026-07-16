"""
Integration Tests

Tests the complete pipeline from SOP to trained model.
"""

import json
import tempfile
import pytest
from pathlib import Path

from src.ecommerce.dataset import (
    SOPBuilder,
    CanonicalCaseGenerator,
    ConversationGenerator,
    DataPipeline,
)


class TestCompletePipeline:
    """Test complete data pipeline."""

    def test_end_to_end_pipeline(self):
        """Test full pipeline from SOP to train/dev/test split."""
        # Step 1: Build SOPs
        builder = SOPBuilder()
        policies = builder.build_fictional_store_sops()
        policies_dict = [p.to_dict() for p in policies]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            policies_path = f.name
            json.dump(policies_dict, f)
        
        # Step 2: Generate canonical cases
        generator = CanonicalCaseGenerator()
        cases = generator.generate_all()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            cases_path = f.name
        
        generator.save_cases(cases_path)
        
        # Step 3: Generate conversations
        with open(policies_path, 'r') as f:
            policies_for_gen = json.load(f)
        
        conv_gen = ConversationGenerator(policies=policies_for_gen, seed=42)
        convs = conv_gen.generate_from_cases(cases, samples_per_case=2)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            convs_path = f.name
        
        conv_gen.save_conversations(convs_path)
        
        # Step 4: Run pipeline
        pipeline = DataPipeline(policies=policies_for_gen, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = pipeline.process(
                input_path=convs_path,
                output_dir=tmpdir,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
            )
            
            # Verify splits
            assert "train" in splits
            assert "dev" in splits
            assert "test" in splits
            
            # Check split files exist
            train_path = Path(tmpdir) / "train.jsonl"
            dev_path = Path(tmpdir) / "dev.jsonl"
            test_path = Path(tmpdir) / "test.jsonl"
            
            assert train_path.exists()
            assert dev_path.exists()
            assert test_path.exists()
            
            # Verify samples have split field
            with open(train_path) as f:
                train_samples = [json.loads(line) for line in f]
            
            assert all(s.get("split") == "train" for s in train_samples)
        
        # Cleanup
        Path(policies_path).unlink(missing_ok=True)
        Path(cases_path).unlink(missing_ok=True)
        Path(convs_path).unlink(missing_ok=True)


class TestPolicyCaseConsistency:
    """Test policy and case consistency."""

    def test_case_policy_ids_exist(self):
        """Test that all case policy references exist."""
        builder = SOPBuilder()
        policies = builder.build_fictional_store_sops()
        policies_dict = [p.to_dict() for p in policies]
        policy_ids = {p["policy_id"] for p in policies_dict}
        
        generator = CanonicalCaseGenerator()
        cases = generator.generate_all()
        
        errors = []
        for case in cases:
            for policy_id in case.expected_policy_ids:
                if policy_id and policy_id not in policy_ids:
                    errors.append(f"Case {case.case_id}: missing policy {policy_id}")
        
        assert len(errors) == 0, f"Missing policies: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
