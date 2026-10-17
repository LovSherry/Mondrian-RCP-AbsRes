from __future__ import annotations

import pandas as pd

from d2c_v2.common import stable_uint64


def _point_xy_km(point_id: str) -> tuple[int, int]:
    value = str(point_id).strip()
    if not value.isdigit() or len(value) != 8:
        raise ValueError(f"invalid LUCAS ETRS-LAEA PointID: {point_id}")
    numeric = int(value)
    return numeric // 10_000, numeric % 10_000


def _block_id(point_id: str, size_km: int) -> str:
    east, north = _point_xy_km(point_id)
    return f"E{east // size_km:03d}_N{north // size_km:03d}_{size_km}km"


def _balanced_fold_map(
    block_counts: pd.Series, folds: int, seed: int
) -> dict[str, int]:
    """Assign intact groups to approximately balanced folds deterministically."""

    items = [
        (str(block), int(count), stable_uint64(str(block), seed))
        for block, count in block_counts.items()
    ]
    items.sort(key=lambda value: (-value[1], value[2]))
    totals = [0] * int(folds)
    mapping: dict[str, int] = {}
    for block, count, digest in items:
        minimum = min(totals)
        choices = [index for index, value in enumerate(totals) if value == minimum]
        fold = choices[digest % len(choices)]
        mapping[block] = fold
        totals[fold] += count
    return mapping
