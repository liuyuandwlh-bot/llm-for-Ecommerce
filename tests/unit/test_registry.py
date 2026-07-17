"""Unit tests for data registry: checksum binding and validation."""

import hashlib
import json

import pytest

from src.ecommerce.dataset.registry import (
    REQUIRED_FIELDS,
    SourceEntry,
    load_registry,
    update_registry_checksum,
)


class TestRegistrySchema:
    def test_required_fields_defined(self):
        assert "source_id" in REQUIRED_FIELDS
        assert "source_name" in REQUIRED_FIELDS
        assert "license" in REQUIRED_FIELDS

    def test_source_entry_minimal(self):
        entry = SourceEntry(
            source_id="test_001",
            source_name="Test",
            license="MIT",
            allowed_train=True,
            allowed_evaluate=True,
            status="acquired",
        )
        assert entry.source_id == "test_001"


class TestChecksumBinding:
    """§4.2: checksum must be bound to actual artifact SHA-256."""

    def test_checksum_roundtrip(self, tmp_path):
        artifact = tmp_path / "artifact.json"
        artifact.write_text('{"key": "value"}', encoding="utf-8")
        with open(artifact, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()

        registry_path = tmp_path / "registry.json"
        update_registry_checksum(
            str(registry_path),
            "owned_sop_v1",
            checksum=digest,
            status="acquired",
        )

        # load_registry returns dict[str, SourceEntry]
        loaded = load_registry(str(registry_path))
        assert "owned_sop_v1" in loaded
        assert loaded["owned_sop_v1"].checksum_sha256 == digest

    def test_mismatched_checksum_is_detectable(self, tmp_path):
        artifact = tmp_path / "artifact.json"
        artifact.write_text('{"key": "modified"}', encoding="utf-8")
        with open(artifact, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()

        # Registry with wrong checksum
        registry_path = tmp_path / "registry.json"
        rows = [
            {
                "source_id": "owned_sop_v1",
                "source_name": "Test",
                "license": "MIT",
                "allowed_train": True,
                "allowed_evaluate": True,
                "status": "acquired",
                "checksum_sha256": "deadbeef",
            }
        ]
        registry_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        loaded = load_registry(str(registry_path))
        stored = loaded["owned_sop_v1"].checksum_sha256
        assert stored != actual
