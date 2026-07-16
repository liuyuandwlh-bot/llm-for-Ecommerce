"""
Hash utilities for deterministic cross-process IDs.

These never use Python's built-in ``hash()`` because its seed is randomized
per interpreter. ``Hashing.digest`` returns a stable SHA-256 hexdigest of
its inputs (joined by ``|``), which is sufficient for fixture-level
templates and split seeding.
"""

import hashlib
import random
from collections.abc import Sequence


class Hashing:
    @staticmethod
    def digest(*parts: object, prefix: str = "") -> str:
        """Return the SHA-256 hexdigest of all parts joined by '|'.

        Args:
            *parts: Strings, ints, or any object with ``__str__``.
            prefix: Optional short string prefix to identify the use.
        """
        joined = "|".join(str(p) for p in parts)
        if prefix:
            joined = f"{prefix}|{joined}"
        return hashlib.sha256(joined.encode('utf-8')).hexdigest()

    @staticmethod
    def short(*parts: object, prefix: str = "", length: int = 12) -> str:
        return Hashing.digest(*parts, prefix=prefix)[:length]

    @staticmethod
    def int(*parts: object, mod: int = 2**31, prefix: str = "seed") -> int:
        """Stable int seed across processes."""
        full = Hashing.digest(*parts, prefix=prefix)
        return int(full[:16], 16) % mod


def seeded_random(*parts: object) -> random.Random:
    """Return a fresh, locally-seeded Random using a stable hash of parts.

    Critical: callers must use this rather than the module-level ``random``
    seed/randint so that different processes produce the same sequences.
    """
    return random.Random(Hashing.int(*parts))


def split_seed(*parts: object) -> int:
    """Stable, non-``hash()`` integer seed."""
    return Hashing.int(*parts, mod=2**31 - 1)


def stable_choice(items: Sequence, *parts: object, default_index: int = 0):
    """Pick an item from ``items`` using a stable hash of the parts.

    Returns ``items[default_index]`` when ``items`` is empty.
    """
    if not items:
        return None
    index = Hashing.int(*parts, mod=len(items))
    return items[index]
