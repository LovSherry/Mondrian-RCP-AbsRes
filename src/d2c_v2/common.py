from __future__ import annotations

import hashlib


def stable_uint64(value: str, seed: int = 0) -> int:
    """Return a platform-independent unsigned hash for deterministic splitting."""

    return int(hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()[:16], 16)
