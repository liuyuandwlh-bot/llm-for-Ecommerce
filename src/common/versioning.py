"""
Versioning Utilities

Track data, model, and code versions for reproducibility.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VersionInfo:
    """Version information for reproducibility."""
    data_hash: str
    model_revision: Optional[str]
    code_commit: str
    config_hash: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "data_hash": self.data_hash,
            "model_revision": self.model_revision,
            "code_commit": self.code_commit,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def compute_data_hash(data_path: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of a data directory or file.

    Args:
        data_path: Path to data file or directory
        algorithm: Hash algorithm (sha256, md5)

    Returns:
        Hex digest of the hash
    """
    path = Path(data_path)

    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if path.is_file():
        _hash_file(hasher, path)
    elif path.is_dir():
        # Hash all files in sorted order for consistency
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith('.'):
                _hash_file(hasher, file_path)

    return hasher.hexdigest()


def _hash_file(hasher, file_path: Path):
    """Hash a single file."""
    # Add relative path to hash for uniqueness
    hasher.update(str(file_path).encode())

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)


def compute_model_hash(model_path: str) -> str:
    """Compute hash of model files."""
    return compute_data_hash(model_path, algorithm="sha256")


def get_git_commit() -> str:
    """Get current git commit SHA."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except:
        return "unknown"


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()


def create_version_info(
    data_path: str,
    model_revision: Optional[str] = None,
    config_path: Optional[str] = None,
) -> VersionInfo:
    """Create version info for current state."""
    data_hash = compute_data_hash(data_path)
    code_commit = get_git_commit()
    timestamp = get_timestamp()

    config_hash = ""
    if config_path:
        config_hash = compute_data_hash(config_path)

    return VersionInfo(
        data_hash=data_hash,
        model_revision=model_revision,
        code_commit=code_commit,
        config_hash=config_hash,
        timestamp=timestamp,
    )
