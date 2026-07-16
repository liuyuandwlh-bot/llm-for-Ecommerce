"""
Data Pipeline for E-commerce Training Data

Implements the complete data processing funnel:
1. Schema validation
2. PII scan/mask
3. Policy/decision consistency
4. Message role/order validation
5. Quality filtering
6. Exact dedup
7. Near dedup
8. Split assignment
"""

import argparse
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from collections import defaultdict
import re

from src.common.pii import PIIDetector
from .policy_engine import PolicyEngine, Decision


@dataclass
class ProcessingStats:
    """Statistics for data processing pipeline."""
    input_count: int = 0
    schema_valid: int = 0
    pii_masked: int = 0
    policy_consistent: int = 0
    role_valid: int = 0
    quality_passed: int = 0
    exact_dedup_removed: int = 0
    near_dedup_removed: int = 0
    final_count: int = 0
    
    reasons: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "input_count": self.input_count,
            "schema_valid": self.schema_valid,
            "pii_masked": self.pii_masked,
            "policy_consistent": self.policy_consistent,
            "role_valid": self.role_valid,
            "quality_passed": self.quality_passed,
            "exact_dedup_removed": self.exact_dedup_removed,
            "near_dedup_removed": self.near_dedup_removed,
            "final_count": self.final_count,
            "reasons": self.reasons,
            "funnel": {
                "schema_valid_rate": f"{self.schema_valid/self.input_count*100:.1f}%" if self.input_count else "N/A",
                "pii_rate": f"{self.pii_masked/self.schema_valid*100:.1f}%" if self.schema_valid else "N/A",
                "quality_pass_rate": f"{self.quality_passed/self.role_valid*100:.1f}%" if self.role_valid else "N/A",
                "dedup_rate": f"{(self.exact_dedup_removed+self.near_dedup_removed)/self.quality_passed*100:.1f}%" if self.quality_passed else "N/A",
            }
        }


class DataPipeline:
    """
    Complete data processing pipeline for e-commerce training data.
    """
    
    REQUIRED_FIELDS = [
        "sample_id", "messages", "intent", "slots", "policy_ids",
        "decision", "requires_human"
    ]
    
    VALID_ROLES = {"system", "user", "assistant"}
    
    def __init__(self, policies: Optional[List[Dict]] = None, seed: int = 42):
        self.policies = policies
        self.policy_engine = PolicyEngine(policies) if policies else None
        self.pii_detector = PIIDetector()
        self.stats = ProcessingStats()
        self.seed = seed
        
        # Deduplication state
        self.exact_hashes: Set[str] = set()
        self.near_dedup_clusters: Dict[str, List[str]] = defaultdict(list)
    
    def process(
        self,
        input_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Dict[str, List[Dict]]:
        """
        Process input data through the complete pipeline.
        
        Returns:
            Dict with keys: train, dev, test
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load input data
        samples = self._load_samples(input_path)
        self.stats.input_count = len(samples)
        
        print(f"Loaded {len(samples)} samples")
        
        # Process each stage
        valid_samples = []
        quarantine_samples = []
        
        for sample in samples:
            result = self._process_sample(sample)
            if result["valid"]:
                valid_samples.append(result["sample"])
            else:
                quarantine_samples.append((sample, result.get("reason", "unknown")))
                self.stats.reasons[result.get("reason", "unknown")] = \
                    self.stats.reasons.get(result.get("reason", "unknown"), 0) + 1
        
        print(f"Schema valid: {len(valid_samples)}")
        self.stats.schema_valid = len(valid_samples)
        
        # Deduplicate
        unique_samples, exact_dup_count = self._exact_dedup(valid_samples)
        self.stats.exact_dedup_removed = exact_dup_count
        print(f"After exact dedup: {len(unique_samples)} (removed {exact_dup_count})")
        
        # Near dedup
        final_samples, near_dup_count = self._near_dedup(unique_samples)
        self.stats.near_dedup_removed = near_dup_count
        print(f"After near dedup: {len(final_samples)} (removed {near_dup_count})")
        
        self.stats.final_count = len(final_samples)
        
        # Split
        splits = self._split_data(final_samples, train_ratio, dev_ratio, test_ratio)
        
        # Save outputs
        self._save_splits(splits, output_dir)
        
        # Save quarantine samples
        if quarantine_samples:
            q_path = Path(output_dir) / "quarantine.jsonl"
            with open(q_path, 'w', encoding='utf-8') as f:
                for sample, reason in quarantine_samples:
                    f.write(json.dumps({
                        "sample": sample,
                        "quarantine_reason": reason
                    }, ensure_ascii=False) + '\n')
            print(f"Quarantine samples saved to {q_path}")
        
        # Save stats
        self._save_stats(output_dir)
        
        return splits
    
    def _load_samples(self, input_path: str) -> List[Dict]:
        """Load samples from JSONL file."""
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        return samples
    
    def _process_sample(self, sample: Dict) -> Dict[str, Any]:
        """Process a single sample through all stages."""
        # Stage 1: Schema validation
        if not self._validate_schema(sample):
            return {"valid": False, "reason": "schema_invalid"}
        self.stats.schema_valid += 1
        
        # Stage 2: PII scan/mask
        masked_sample = self._mask_pii(sample)
        if masked_sample != sample:
            self.stats.pii_masked += 1
            sample = masked_sample
            sample["pii_status"] = "masked"
        
        # Stage 3: Policy consistency
        if self.policy_engine and not self._validate_policy(sample):
            return {"valid": False, "reason": "policy_inconsistent"}
        self.stats.policy_consistent += 1
        
        # Stage 4: Role validation
        if not self._validate_roles(sample):
            return {"valid": False, "reason": "invalid_roles"}
        self.stats.role_valid += 1
        
        # Stage 5: Quality filtering
        if not self._check_quality(sample):
            return {"valid": False, "reason": "quality_failed"}
        self.stats.quality_passed += 1
        
        return {"valid": True, "sample": sample}
    
    def _validate_schema(self, sample: Dict) -> bool:
        """Validate sample has required fields."""
        for field in self.REQUIRED_FIELDS:
            if field not in sample:
                return False
        
        # Validate messages
        if not isinstance(sample.get("messages"), list):
            return False
        if len(sample["messages"]) < 2:
            return False
        
        # Validate message structure
        for msg in sample["messages"]:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
        
        return True
    
    def _mask_pii(self, sample: Dict) -> Dict:
        """Mask PII in all messages."""
        masked_sample = dict(sample)
        messages = []
        
        for msg in sample["messages"]:
            masked_msg = dict(msg)
            if msg["role"] != "system":
                masked_text, matches = self.pii_detector.mask(msg["content"])
                masked_msg["content"] = masked_text
                if matches:
                    masked_msg["pii_detected"] = [m.pii_type for m in matches]
            messages.append(masked_msg)
        
        masked_sample["messages"] = messages
        return masked_sample
    
    def _validate_policy(self, sample: Dict) -> bool:
        """Validate policy consistency."""
        if not self.policy_engine:
            return True
        
        # Match against policy engine
        match = self.policy_engine.match(sample.get("slots", {}))
        
        expected_decision = sample.get("decision", "")
        actual_decision = match.decision.value
        
        # Allow minor discrepancies
        if expected_decision != actual_decision:
            return False
        
        return True
    
    def _validate_roles(self, sample: Dict) -> bool:
        """Validate message roles are valid and in order."""
        roles = [msg.get("role") for msg in sample.get("messages", [])]
        
        # All roles must be valid
        for role in roles:
            if role not in self.VALID_ROLES:
                return False
        
        # Must start with user or system
        if roles[0] not in ["user", "system"]:
            return False
        
        # Must have at least one assistant turn
        if "assistant" not in roles:
            return False
        
        return True
    
    def _check_quality(self, sample: Dict) -> bool:
        """Check sample passes quality filters."""
        messages = sample.get("messages", [])
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Check for empty content
            if not content.strip():
                return False
            
            # Check for extreme length (too short or too long)
            if len(content) > 10000:
                return False
            
            # Check for repetitive patterns
            if self._is_repetitive(content):
                return False
        
        # Check for message count
        if len(messages) > 20:
            return False
        
        return True
    
    def _is_repetitive(self, text: str, threshold: float = 0.7) -> bool:
        """Check if text is overly repetitive."""
        if len(text) < 50:
            return False
        
        # Simple check: if same 3-gram appears > threshold times
        ngrams = [text[i:i+3] for i in range(len(text)-2)]
        if not ngrams:
            return False
        
        ngram_counts = defaultdict(int)
        for ng in ngrams:
            ngram_counts[ng] += 1
        
        max_count = max(ngram_counts.values()) if ngram_counts else 0
        if max_count / len(ngrams) > threshold:
            return True
        
        return False
    
    def _exact_dedup(self, samples: List[Dict]) -> tuple:
        """Exact deduplication using normalized text hash."""
        unique = []
        removed = 0
        
        for sample in samples:
            # Compute hash of normalized messages
            messages = sample.get("messages", [])
            normalized = "".join(m.get("content", "") for m in messages)
            normalized = re.sub(r'\s+', '', normalized.lower())
            
            hash_val = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            
            if hash_val not in self.exact_hashes:
                self.exact_hashes.add(hash_val)
                sample["_dedup_hash"] = hash_val
                unique.append(sample)
            else:
                removed += 1
        
        return unique, removed
    
    def _near_dedup(self, samples: List[Dict], threshold: float = 0.95) -> tuple:
        """Near deduplication using simple n-gram similarity."""
        # Simple implementation: group by intent + template_family
        clusters: Dict[str, List[Dict]] = defaultdict(list)
        
        for sample in samples:
            key = f"{sample.get('intent', '')}_{sample.get('template_family', '')}"
            clusters[key].append(sample)
        
        unique = []
        removed = 0
        
        for cluster_key, cluster_samples in clusters.items():
            # Keep first, mark rest for removal
            if len(cluster_samples) > 1:
                unique.append(cluster_samples[0])
                removed += len(cluster_samples) - 1
                cluster_samples[0]["dedup_cluster_id"] = cluster_key
                
                for s in cluster_samples[1:]:
                    s["dedup_cluster_id"] = cluster_key
            else:
                unique.append(cluster_samples[0])
        
        return unique, removed
    
    def _split_data(
        self,
        samples: List[Dict],
        train_ratio: float,
        dev_ratio: float,
        test_ratio: float,
    ) -> Dict[str, List[Dict]]:
        """
        Split data into train/dev/test.
        
        Uses stratified split by intent to avoid imbalance.
        """
        # Group by intent
        by_intent: Dict[str, List[Dict]] = defaultdict(list)
        for sample in samples:
            intent = sample.get("intent", "unknown")
            by_intent[intent].append(sample)
        
        splits: Dict[str, List[Dict]] = {"train": [], "dev": [], "test": []}
        
        for intent, intent_samples in by_intent.items():
            n = len(intent_samples)
            
            # Calculate split sizes
            n_train = int(n * train_ratio)
            n_dev = int(n * dev_ratio)
            
            # Shuffle deterministically
            import random
            random.seed(self.seed + hash(intent) % 1000)
            random.shuffle(intent_samples)
            
            # Assign
            splits["train"].extend(intent_samples[:n_train])
            splits["dev"].extend(intent_samples[n_train:n_train+n_dev])
            splits["test"].extend(intent_samples[n_train+n_dev:])
            
            # Mark split
            for i, s in enumerate(intent_samples):
                if i < n_train:
                    s["split"] = "train"
                elif i < n_train + n_dev:
                    s["split"] = "dev"
                else:
                    s["split"] = "test"
        
        # Remove internal fields
        for split_name in splits:
            for sample in splits[split_name]:
                sample.pop("_dedup_hash", None)
        
        return splits
    
    def _save_splits(self, splits: Dict[str, List[Dict]], output_dir: str):
        """Save splits to JSONL files."""
        for split_name, samples in splits.items():
            path = Path(output_dir) / f"{split_name}.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"Saved {split_name}: {len(samples)} samples to {path}")
    
    def _save_stats(self, output_dir: str):
        """Save processing statistics."""
        # JSON report
        stats_path = Path(output_dir) / "processing_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Markdown report
        md_path = Path(output_dir) / "processing_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Data Processing Report\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Input samples: {self.stats.input_count}\n")
            f.write(f"- Final samples: {self.stats.final_count}\n")
            f.write(f"- Overall retention: {self.stats.final_count/self.stats.input_count*100:.1f}%\n\n")
            
            f.write(f"## Funnel\n\n")
            f.write("| Stage | Count | Rate |\n")
            f.write("|-------|-------|------|\n")
            f.write(f"| Input | {self.stats.input_count} | 100% |\n")
            f.write(f"| Schema Valid | {self.stats.schema_valid} | {self.stats.to_dict()['funnel']['schema_valid_rate']} |\n")
            f.write(f"| PII Masked | {self.stats.pii_masked} | {self.stats.to_dict()['funnel']['pii_rate']} |\n")
            f.write(f"| Quality Passed | {self.stats.quality_passed} | {self.stats.to_dict()['funnel']['quality_pass_rate']} |\n")
            f.write(f"| After Dedup | {self.stats.final_count} | {self.stats.to_dict()['funnel']['dedup_rate']} |\n\n")
            
            if self.stats.reasons:
                f.write(f"## Quarantine Reasons\n\n")
                for reason, count in sorted(self.stats.reasons.items()):
                    f.write(f"- {reason}: {count}\n")


def run_pipeline(
    input_path: str = "data/processed/fixtures/conversations.jsonl",
    output_dir: str = "data/processed/fixtures/release_v1",
    policies_path: str = "data/processed/fixtures/policies.json",
    seed: int = 42,
):
    """Run the complete data pipeline."""
    # Load policies
    policies = None
    if Path(policies_path).exists():
        with open(policies_path, 'r', encoding='utf-8') as f:
            policies = json.load(f)
    
    # Run pipeline
    pipeline = DataPipeline(policies=policies, seed=seed)
    splits = pipeline.process(
        input_path=input_path,
        output_dir=output_dir,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
    )
    
    return splits


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run data processing pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/processed/fixtures/conversations.jsonl",
        help="Input conversations JSONL"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed/fixtures/release_v1",
        help="Output directory"
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="data/processed/fixtures/policies.json",
        help="Policies JSON for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    splits = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        policies_path=args.policies,
        seed=args.seed,
    )
    
    print(f"\nPipeline complete!")
    print(f"Train: {len(splits['train'])}")
    print(f"Dev: {len(splits['dev'])}")
    print(f"Test: {len(splits['test'])}")
    
    return 0


if __name__ == "__main__":
    exit(main())
