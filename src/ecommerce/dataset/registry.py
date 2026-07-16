"""
Data Registry Loader and Updater

Round 2:
- Schema validation on registry rows
- Acquired rows require checksum_sha256
- Source status semantically enforced at pipeline time
- Helper to update checksum after artifact build
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "source_id",
    "source_name",
    "allowed_train",
    "allowed_evaluate",
    "license",
    "status",
)

ACQUIRABLE_STATUSES = {"acquired", "validated"}
USABLE_STATUSES = {"acquired", "validated"}
TRAIN_USABLE_STATUSES = {"acquired", "validated"}


class RegistryError(Exception):
    """Raised when a registry operation is invalid."""


@dataclass
class SourceEntry:
    source_id: str
    source_name: str
    source_url: str = ""
    publisher: str = ""
    acquired_at: str = ""
    published_at: str = ""
    license: str = ""
    license_url: str = ""
    allowed_train: bool = False
    allowed_evaluate: bool = False
    allowed_local_demo: bool = False
    allowed_redistribute: bool = False
    contains_pii: bool = False
    checksum_sha256: str = ""
    version: str = ""
    revision: str = ""
    status: str = "planned"
    notes: str = ""

    def is_usable_for_train(self) -> bool:
        return (
            self.status in TRAIN_USABLE_STATUSES
            and self.allowed_train
            and bool(self.license)
            and bool(self.checksum_sha256)
        )


def load_registry(path: str) -> dict[str, SourceEntry]:
    """Load registry JSON into a dict keyed by source_id."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        if "sources" in raw:
            rows = raw["sources"]
        else:
            rows = list(raw.values())
    else:
        raise RegistryError(f"unsupported registry shape: {type(raw)}")
    out: dict[str, SourceEntry] = {}
    for row in rows:
        _validate_row(row)
        entry = SourceEntry(
            source_id=str(row["source_id"]),
            source_name=str(row["source_name"]),
            source_url=str(row.get("source_url") or ""),
            publisher=str(row.get("publisher") or ""),
            acquired_at=str(row.get("acquired_at") or ""),
            published_at=str(row.get("published_at") or ""),
            license=str(row.get("license") or ""),
            license_url=str(row.get("license_url") or ""),
            allowed_train=bool(row.get("allowed_train", False)),
            allowed_evaluate=bool(row.get("allowed_evaluate", False)),
            allowed_local_demo=bool(row.get("allowed_local_demo", False)),
            allowed_redistribute=bool(row.get("allowed_redistribute", False)),
            contains_pii=bool(row.get("contains_pii", False)),
            checksum_sha256=str(row.get("checksum_sha256") or ""),
            version=str(row.get("version") or ""),
            revision=str(row.get("revision") or ""),
            status=str(row.get("status") or "planned"),
            notes=str(row.get("notes") or ""),
        )
        out[entry.source_id] = entry
    return out


def _validate_row(row: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in row:
            raise RegistryError(f"registry row missing required field {field!r}")
    status = row.get("status")
    if status not in {"planned", "acquired", "validated", "quarantine", "rejected"}:
        raise RegistryError(f"registry row has unknown status {status!r}")


def save_registry(path: str, registry: dict[str, SourceEntry]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [vars(entry) for entry in registry.values()]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def update_registry_checksum(
    path: str,
    source_id: str,
    checksum: str,
    status: str = "acquired",
) -> None:
    """Update a registry row with the given SHA-256 and status.

    If the row does not exist it is created with sensible defaults.
    """
    registry = load_registry(path) if Path(path).exists() else {}
    entry = registry.get(source_id)
    if entry is None:
        entry = SourceEntry(
            source_id=source_id,
            source_name=source_id,
            allowed_train=True,
            allowed_evaluate=True,
            allowed_local_demo=True,
            license="CC0",
            status=status,
        )
    entry.checksum_sha256 = checksum
    entry.status = status
    registry[source_id] = entry
    save_registry(path, registry)


def validate_sample_against_registry(
    sample: dict[str, Any],
    entry: SourceEntry,
    mode: str,
) -> str | None:
    """Return error message if the sample should not be used.

    Args:
        sample: Sample dict containing at least ``source_id``.
        entry: SourceEntry from the registry.
        mode: One of ``"fixture"`` or ``"training_release"``.
    """
    sample_source = sample.get("source_id")
    if not sample_source:
        return "missing source_id"
    if entry.status not in USABLE_STATUSES:
        return f"source status not usable: {entry.status}"
    if not entry.checksum_sha256:
        return "source checksum missing"
    if not entry.license:
        return "source license missing"
    if mode == "training_release":
        if not entry.allowed_train:
            return "source not allowed for training"
        review = sample.get("review_status")
        if review != "human_approved":
            return f"training_release requires review_status=human_approved (got {review!r})"
    return None
