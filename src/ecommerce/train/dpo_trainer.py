"""
DPO Training - Round 2 placeholder.

This module exists for documentation purposes only. DPO requires a real
preference dataset that we have not collected yet, so this build refuses to
train and exposes a strict data-schema validator instead.

CLI usage:

    python -m src.ecommerce.train.dpo_trainer --validate-data <path.jsonl>
    python -m src.ecommerce.train.dpo_trainer --check-schema
    python -m src.ecommerce.train.dpo_trainer  # prints experimental notice

Until a real human preference dataset is provided, do NOT enable DPO
training. The README documents this explicitly.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_KEYS = ("prompt", "chosen", "rejected")
ALLOWED_KEYS = set(REQUIRED_KEYS) | {"intent", "annotator", "annotation_date"}


class DPOUnavailableError(RuntimeError):
    """Raised when DPO is invoked without a verified preference dataset."""


@dataclass
class DPOPairSchema:
    prompt: str
    chosen: str
    rejected: str
    intent: str = ""
    annotator: str = ""
    annotation_date: str = ""


def validate_pair(record: dict[str, Any]) -> list[str]:
    """Validate one preference pair record. Return list of error messages."""
    errors: list[str] = []
    missing = [k for k in REQUIRED_KEYS if k not in record]
    if missing:
        errors.append(f"missing required keys: {missing}")
        return errors
    for k in REQUIRED_KEYS:
        if not isinstance(record[k], str) or not record[k].strip():
            errors.append(f"{k!r} must be non-empty string")
    unknown = set(record.keys()) - ALLOWED_KEYS
    if unknown:
        errors.append(f"unknown keys: {sorted(unknown)}")
    if record.get("chosen") == record.get("rejected"):
        errors.append("chosen and rejected are identical; pair is degenerate")
    return errors


def validate_pairs_file(path: str) -> dict[str, Any]:
    n_total = 0
    n_valid = 0
    issues: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({"line": line_no, "error": f"json: {exc}"})
                continue
            errs = validate_pair(rec)
            if errs:
                issues.append({"line": line_no, "errors": errs})
            else:
                n_valid += 1
    return {
        "total": n_total,
        "valid": n_valid,
        "issues": issues[:20],
        "ok": (n_valid > 0 and not issues),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DPO trainer (Round 2 placeholder)")
    parser.add_argument(
        "--check-schema", action="store_true", help="Print required schema for preference pairs."
    )
    parser.add_argument(
        "--validate-data", type=str, help="Validate a JSONL preference dataset without training."
    )
    args = parser.parse_args()

    if args.check_schema:
        print(
            json.dumps(
                {
                    "required_keys": list(REQUIRED_KEYS),
                    "optional_keys": ["intent", "annotator", "annotation_date"],
                    "experimental": True,
                    "training_enabled": False,
                    "note": (
                        "DPO training requires a verified human preference dataset. "
                        "Until then this CLI only validates data schema."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.validate_data:
        if not Path(args.validate_data).exists():
            print(f"file not found: {args.validate_data}")
            return 1
        report = validate_pairs_file(args.validate_data)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 2

    # Default: refuse to run.
    print(
        "DPO is intentionally disabled in this Round 2 build. "
        "Use --check-schema to print the required schema, or --validate-data "
        "<path> to validate a preference dataset. "
        "Real DPO training will be re-enabled once a human preference "
        "dataset is available."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
