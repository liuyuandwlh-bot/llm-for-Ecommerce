"""
Data Pipeline for E-commerce Training Data

Round 2:
- Per-stage funnel counts (no rolling counters)
- First-failure attribution per sample
- Real registry validation (status/license/checksum)
- Real n-gram Jaccard near-dedup
- Group-aware stratified split with leakage report and cross-process
  determinism via stable hashing.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.hashing import Hashing
from src.common.near_dedup import near_dedup
from src.common.pii import PIIDetector

from .policy_engine import PolicyEngine
from .registry import (
    RegistryError,
    load_registry,
    validate_sample_against_registry,
)

# Stage order matters: a sample advances only if the previous stage passed.
STAGES = (
    "input",
    "schema_pass",
    "registry_pass",
    "pii_masked",
    "policy_pass",
    "role_pass",
    "quality_pass",
    "review_pass",
    "exact_dedup_removed",
    "near_dedup_removed",
    "final",
)


# ---------------------------------------------------------------------------
# Stats / reporting
# ---------------------------------------------------------------------------


@dataclass
class StageStats:
    """Funnel statistics where each entry counts the *cumulative* survivors
    of that stage (i.e. those samples that passed every prior stage).
    """

    counts: dict[str, int] = field(default_factory=dict)
    rejections: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for stage in STAGES:
            self.counts.setdefault(stage, 0)

    def to_dict(self) -> dict:
        return {
            "stage_counts": self.counts,
            "rejection_summary": _summary_rejections(self.rejections),
        }


def _summary_rejections(rejections: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = defaultdict(int)
    for r in rejections:
        summary[r["stage"]] += 1
    return dict(summary)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


ALLOWED_ROLES = {"system", "user", "assistant"}


def validate_schema(sample: dict[str, Any]) -> str | None:
    """Return None if OK, else error message."""
    for required in ("sample_id", "messages", "intent", "decision", "policy_ids"):
        if required not in sample:
            return f"missing field {required!r}"
    if not isinstance(sample["messages"], list) or not sample["messages"]:
        return "messages must be non-empty list"
    if not isinstance(sample["policy_ids"], list):
        return "policy_ids must be list"
    return None


def validate_role_order(sample: dict[str, Any]) -> str | None:
    """Validate role sequence: starts with system or user; user/assistant alternation."""
    messages = sample["messages"]
    first = messages[0].get("role")
    if first not in {"system", "user"}:
        return f"first message must be system or user, got {first!r}"
    last_role = None
    saw_user = False
    for m in messages:
        role = m.get("role")
        if role not in ALLOWED_ROLES:
            return f"invalid role {role!r}"
        if role == "user":
            saw_user = True
        if last_role == "assistant" and role == "assistant":
            return "consecutive assistant messages"
        if role != "system" and role == last_role:
            return f"non-system role {role!r} repeats"
        last_role = role
    if not saw_user:
        return "no user turn"
    if last_role != "assistant":
        return "last message must be assistant"
    return None


def validate_review(sample: dict[str, Any], mode: str) -> str | None:
    review = sample.get("review_status", "pending")
    if mode == "fixture":
        if review not in {"auto_validated", "human_approved"}:
            return f"fixture mode requires review_status in {{auto_validated, human_approved}}, got {review!r}"
    elif mode == "training_release":
        if review != "human_approved":
            return f"training_release requires human_approved, got {review!r}"
    return None


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


DEFAULT_SPLIT_RATIOS = {"train": 0.7, "dev": 0.15, "test": 0.15}


def _group_id(sample: dict[str, Any]) -> str:
    """Group sample by union of (parent_case_id, template_family).

    Note: we deliberately omit ``dedup_cluster_id`` from the group key, so
    all rewrite strategies of the same canonical case remain in the same
    group and never leak across splits.
    """
    parts = [
        sample.get("parent_case_id", ""),
        sample.get("template_family", ""),
    ]
    if not any(parts):
        return "group_" + Hashing.short("group", sample.get("sample_id", ""), length=16)
    return "g_" + Hashing.short("group", *parts, length=20)


def stratified_group_split(
    samples: list[dict[str, Any]],
    seed: int,
    ratios: dict[str, float] = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Split samples by group, stratifying by intent.

    Algorithm:
    - Group samples by ``_group_id(sample)``.
    - Sort groups by stable (intent, group_id) and walk through them,
      placing each entire group into the smallest split first, subject to
      the ratio (with a small tolerance).
    - Within a split, sort samples by ``sample_id`` for stable output order.
    """
    if ratios is None:
        ratios = DEFAULT_SPLIT_RATIOS
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {total}")

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    intent_of_group: dict[str, str] = {}
    for sample in samples:
        gid = _group_id(sample)
        groups[gid].append(sample)
        intent_of_group.setdefault(gid, sample.get("intent", "unknown"))

    # Order groups by stable (intent, group_id) so it's deterministic.
    sorted_groups = sorted(
        groups.keys(),
        key=lambda g: (intent_of_group[g], g),
    )

    # Allocate targets by ratio of total groups
    n = len(sorted_groups)
    if n < 3:
        target = {"train": max(1, n - 2), "dev": 1, "test": 1}
    else:
        # Reserve dev and test using the requested ratios; the rest go to train.
        target_dev = max(1, int(round(ratios["dev"] * n)))
        target_test = max(1, int(round(ratios["test"] * n)))
        if target_dev + target_test >= n:
            target_dev = max(1, n // 3)
            target_test = max(1, n // 3)
        target = {"train": n - target_dev - target_test, "dev": target_dev, "test": target_test}

    # Track intent counts per split to balance stratification.
    intent_counts = {k: defaultdict(int) for k in splits}

    placed: dict[str, str] = {}
    ordered_groups = list(sorted_groups)
    target_dev = target["dev"]
    target_test = target["test"]

    # Interleave dev/test picks by alternating across the stable order so each
    # non-train split gets balanced intent coverage. Remaining groups go to
    # train so that train is by construction the largest split.
    dev_picks: list[str] = []
    test_picks: list[str] = []
    for i, gid in enumerate(ordered_groups):
        if len(dev_picks) < target_dev and i % 2 == 0:
            dev_picks.append(gid)
        elif len(test_picks) < target_test and i % 2 == 1:
            test_picks.append(gid)
        elif len(dev_picks) < target_dev:
            dev_picks.append(gid)
        elif len(test_picks) < target_test:
            test_picks.append(gid)

    for gid in dev_picks:
        splits["dev"].extend(groups[gid])
        intent_counts["dev"][intent_of_group[gid]] += 1
        placed[gid] = "dev"
    for gid in test_picks:
        if gid in placed:
            continue
        splits["test"].extend(groups[gid])
        intent_counts["test"][intent_of_group[gid]] += 1
        placed[gid] = "test"
    for gid in ordered_groups:
        if gid in placed:
            continue
        splits["train"].extend(groups[gid])
        intent_counts["train"][intent_of_group[gid]] += 1
        placed[gid] = "train"

    # Sort samples inside each split by sample_id for stable ordering.
    for k in splits:
        splits[k] = sorted(splits[k], key=lambda s: s["sample_id"])

    manifest = {
        "algorithm": "stratified_group_v1",
        "seed": seed,
        "ratios": ratios,
        "total_groups": len(groups),
        "total_samples": len(samples),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_group_counts": {
            k: sum(1 for s in v for gid in [placed.get(_group_id(s))]
                   if gid)
            for k, v in splits.items()
        },
        "split_intent_counts": {
            k: dict(intent_counts[k]) for k in splits
        },
    }
    return splits, manifest


def check_leakage(
    splits: dict[str, list[dict[str, Any]]],
) -> tuple[bool, list[dict[str, Any]]]:
    """Verify that no group / dedup_cluster_id appears in more than one split."""
    group_to_split: dict[str, str] = {}
    cluster_to_split: dict[str, str] = {}
    leaks: list[dict[str, Any]] = []
    for split_name, samples in splits.items():
        for s in samples:
            gid = _group_id(s)
            cid = s.get("dedup_cluster_id")
            if gid in group_to_split and group_to_split[gid] != split_name:
                leaks.append({"type": "group", "id": gid, "splits": [group_to_split[gid], split_name]})
            group_to_split[gid] = split_name
            if cid:
                if cid in cluster_to_split and cluster_to_split[cid] != split_name:
                    leaks.append({"type": "dedup_cluster", "id": cid, "splits": [cluster_to_split[cid], split_name]})
                cluster_to_split[cid] = split_name
    return (len(leaks) == 0, leaks)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    near_dedup_threshold: float = 0.7
    seed: int = 42
    mode: str = "fixture"  # "fixture" or "training_release"
    split_ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS))


def run_pipeline(
    conversations_path: str,
    policies_path: str,
    registry_path: str,
    output_dir: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Run the full data funnel. Returns a structured report dict."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(policies_path, encoding='utf-8') as f:
        policies = json.load(f)
    engine = PolicyEngine(policies)

    if Path(registry_path).exists():
        try:
            registry = load_registry(registry_path)
        except Exception as exc:
            raise RegistryError(f"failed to load registry: {exc}") from exc
    else:
        registry = {}

    samples_in = _load_jsonl(conversations_path)
    stats = StageStats()
    stats.counts["input"] = len(samples_in)

    survived: list[dict[str, Any]] = []
    for sample in samples_in:
        err = validate_schema(sample)
        if err:
            stats.rejections.append({"sample_id": sample.get("sample_id", "<unknown>"), "stage": "schema_pass", "reason": err})
            continue
        stats.counts["schema_pass"] += 1

        source_id = sample.get("source_id", "")
        entry = registry.get(source_id)
        if entry is None:
            stats.rejections.append({"sample_id": sample["sample_id"], "stage": "registry_pass", "reason": f"unknown source {source_id!r}"})
            continue
        err = validate_sample_against_registry(sample, entry, mode=config.mode)
        if err:
            stats.rejections.append({"sample_id": sample["sample_id"], "stage": "registry_pass", "reason": err})
            continue
        stats.counts["registry_pass"] += 1

        # PII scan + mask
        pii = PIIDetector()
        any_pii = False
        for msg in sample["messages"]:
            text = msg.get("content", "")
            masked, spans = pii.mask(text)
            if spans:
                any_pii = True
                msg["content"] = masked
        if any_pii:
            sample["pii_status"] = "masked"
        else:
            sample["pii_status"] = "passed"
        stats.counts["pii_masked"] += 1

        # Policy consistency (only when expected_policy_ids is non-empty)
        if sample.get("policy_ids"):
            match = engine.match(
                sample.get("slots") or {},
                category_hint=sample.get("category_hint"),
            )
            if match.policy_id and match.policy_id not in sample["policy_ids"]:
                stats.rejections.append({"sample_id": sample["sample_id"], "stage": "policy_pass", "reason": f"policy mismatch with engine: {match.policy_id}", "kind": "policy_inconsistent"})
                continue
            if match.decision.value != sample.get("decision"):
                stats.rejections.append({"sample_id": sample["sample_id"], "stage": "policy_pass", "reason": f"decision mismatch: engine={match.decision.value} sample={sample.get('decision')}", "kind": "policy_inconsistent"})
                continue
        stats.counts["policy_pass"] += 1

        err = validate_role_order(sample)
        if err:
            stats.rejections.append({"sample_id": sample["sample_id"], "stage": "role_pass", "reason": err, "kind": "invalid_roles"})
            continue
        stats.counts["role_pass"] += 1

        # Quality (placeholder for richer rules; reject empty user/assistant content)
        quality_ok = True
        for msg in sample["messages"]:
            content = (msg.get("content") or "").strip()
            if not content:
                quality_ok = False
                break
        if not quality_ok:
            stats.rejections.append({"sample_id": sample["sample_id"], "stage": "quality_pass", "reason": "empty content"})
            continue
        stats.counts["quality_pass"] += 1

        # Review mode gate
        err = validate_review(sample, config.mode)
        if err:
            stats.rejections.append({"sample_id": sample["sample_id"], "stage": "review_pass", "reason": err})
            continue
        stats.counts["review_pass"] += 1

        survived.append(sample)

    # Exact dedup by canonicalized user text
    seen: dict[str, str] = {}
    survivors: list[dict[str, Any]] = []
    for sample in survived:
        from src.common.near_dedup import normalize_text
        text = normalize_text(_user_business_text(sample))
        if not text:
            survivors.append(sample)
            continue
        if text in seen:
            stats.rejections.append({
                "sample_id": sample["sample_id"],
                "stage": "exact_dedup_removed",
                "reason": f"duplicate of {seen[text]}",
            })
            stats.counts["exact_dedup_removed"] += 1
            continue
        seen[text] = sample["sample_id"]
        survivors.append(sample)

    # Near dedup
    near_result = near_dedup(survivors, threshold=config.near_dedup_threshold)
    near_removed_ids = {item["sample_id"] for item in near_result.near_removed}
    near_quarantined_ids = {item["sample_id"] for item in near_result.quarantined}

    kept_for_split: list[dict[str, Any]] = []
    for s in survivors:
        if s["sample_id"] in near_removed_ids or s["sample_id"] in near_quarantined_ids:
            stats.rejections.append({
                "sample_id": s["sample_id"],
                "stage": "near_dedup_removed",
                "reason": "near_dup_quarantine" if s["sample_id"] in near_quarantined_ids else "near_dup",
            })
            stats.counts["near_dedup_removed"] += 1
            continue
        kept_for_split.append(s)

    # Compute final counts
    splits, split_manifest = stratified_group_split(
        kept_for_split,
        seed=config.seed,
        ratios=config.split_ratios,
    )

    splits_total = sum(len(v) for v in splits.values())
    stats.counts["final"] = splits_total

    # Write splits
    _write_jsonl(Path(output_dir) / "train.jsonl", splits["train"])
    _write_jsonl(Path(output_dir) / "dev.jsonl", splits["dev"])
    _write_jsonl(Path(output_dir) / "test.jsonl", splits["test"])
    _write_jsonl(Path(output_dir) / "quarantine.jsonl", [
        {**next((s for s in survivors if s["sample_id"] == item["sample_id"]), item), **{"drop_reason": item.get("reason", "")}}
        for item in near_result.quarantined
    ])

    # Leakage
    leak_ok, leaks = check_leakage(splits)
    leakage_report = {
        "no_leakage": leak_ok,
        "leak_count": len(leaks),
        "leaks": leaks[:20],  # cap for visibility
    }
    with open(Path(output_dir) / "leakage_report.json", 'w', encoding='utf-8') as f:
        json.dump(leakage_report, f, ensure_ascii=False, indent=2)
    # Markdown summary
    md_lines = [
        "# Leakage Report",
        "",
        f"- no_leakage: {leak_ok}",
        f"- leak_count: {len(leaks)}",
    ]
    if leaks:
        md_lines.append("\n## Sample leaks (first 20)\n")
        for lk in leaks[:20]:
            md_lines.append(f"- {lk['type']} {lk['id']} spans {lk['splits']}")
    with open(Path(output_dir) / "leakage_report.md", 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")

    # Manifest
    manifest = {
        "algorithm_version": split_manifest["algorithm"],
        "seed": split_manifest["seed"],
        "ratios": split_manifest["ratios"],
        "input_hash": _hash_jsonl(conversations_path),
        "total_samples": split_manifest["total_samples"],
        "total_groups": split_manifest["total_groups"],
        "splits": {
            k: {
                "sample_ids": [s["sample_id"] for s in v],
                "group_ids": sorted({_group_id(s) for s in v}),
                "intent_counts": dict(split_manifest["split_intent_counts"][k]),
            }
            for k, v in splits.items()
        },
    }
    with open(Path(output_dir) / "split_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Funnel report
    funnel = {
        "stage_counts": stats.counts,
        "rejection_summary": _summary_rejections(stats.rejections),
        "invalid_roles": sum(1 for r in stats.rejections if r.get("kind") == "invalid_roles"),
        "policy_inconsistent": sum(1 for r in stats.rejections if r.get("kind") == "policy_inconsistent"),
        "near_dedup": near_result.stats,
        "leakage": {"no_leakage": leak_ok, "leak_count": len(leaks)},
        "split_manifest": manifest,
    }
    with open(Path(output_dir) / "funnel_report.json", 'w', encoding='utf-8') as f:
        json.dump(funnel, f, ensure_ascii=False, indent=2)
    with open(Path(output_dir) / "funnel_report.md", 'w', encoding='utf-8') as f:
        f.write("# Funnel Report\n\n")
        f.write("## Stage counts\n\n")
        for stage in STAGES:
            f.write(f"- {stage}: {stats.counts.get(stage, 0)}\n")
        f.write("\n## Rejections\n\n")
        for stage, count in _summary_rejections(stats.rejections).items():
            f.write(f"- {stage}: {count}\n")

    # Exit code: only fail if leakage detected and on training_release
    exit_ok = True
    if not leak_ok and config.mode == "training_release":
        exit_ok = False
    return {"funnel": funnel, "exit_ok": exit_ok}


def _user_business_text(sample: dict[str, Any]) -> str:
    from src.common.near_dedup import extract_user_business_text
    return extract_user_business_text(sample)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def _hash_jsonl(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------


class DataPipeline:  # pragma: no cover - thin wrapper
    """Deprecated alias for the module-level ``run_pipeline`` function.

    Existing tests used ``DataPipeline(...).run()``; keep that working.
    """

    def __init__(
        self,
        policies_path: str,
        registry_path: str,
        output_dir: str,
        seed: int = 42,
        mode: str = "fixture",
        near_dedup_threshold: float = 0.7,
        split_ratios: dict[str, float] | None = None,
    ):
        self.config = PipelineConfig(
            near_dedup_threshold=near_dedup_threshold,
            seed=seed,
            mode=mode,
            split_ratios=split_ratios or dict(DEFAULT_SPLIT_RATIOS),
        )
        self.policies_path = policies_path
        self.registry_path = registry_path
        self.output_dir = output_dir

    def run(self, conversations_path: str) -> dict[str, Any]:
        return run_pipeline(
            conversations_path,
            self.policies_path,
            self.registry_path,
            self.output_dir,
            self.config,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="E-commerce data pipeline")
    parser.add_argument("--input", required=True, help="Conversations JSONL")
    parser.add_argument("--policies", required=True, help="Policies JSON")
    parser.add_argument("--registry", required=True, help="Registry JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["fixture", "training_release"], default="fixture")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dedup-threshold", type=float, default=0.7)
    parser.add_argument("--ratios", type=str, default="0.7,0.15,0.15", help="train,dev,test ratios")
    args = parser.parse_args()

    ratios = {}
    parts = [p.strip() for p in args.ratios.split(",") if p.strip()]
    if len(parts) != 3:
        print("--ratios must be 3 numbers summing to 1")
        return 2
    ratios = {"train": float(parts[0]), "dev": float(parts[1]), "test": float(parts[2])}

    config = PipelineConfig(
        near_dedup_threshold=args.near_dedup_threshold,
        seed=args.seed,
        mode=args.mode,
        split_ratios=ratios,
    )

    try:
        report = run_pipeline(
            args.input,
            args.policies,
            args.registry,
            args.output,
            config,
        )
    except RegistryError as exc:
        print(f"Registry error: {exc}")
        return 1

    funnel = report["funnel"]
    print("Stage counts:")
    for stage in STAGES:
        print(f"  {stage} = {funnel['stage_counts'].get(stage, 0)}")
    print(f"invalid_roles = {funnel['invalid_roles']}")
    print(f"policy_inconsistent = {funnel['policy_inconsistent']}")
    print(f"leak_count = {funnel['leakage']['leak_count']}")
    print("train/dev/test sizes:")
    for k, v in funnel["split_manifest"]["splits"].items():
        print(f"  {k}: {len(v['sample_ids'])}")
    return 0 if report["exit_ok"] else 1


if __name__ == "__main__":
    exit(main())
