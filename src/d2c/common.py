from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


FORBIDDEN_PREDICTORS = {
    "point_id",
    "survey_year",
    "country",
    "spatial_block_id",
    "role",
    "role_code",
    "longitude",
    "latitude",
    "nuts0",
    "nuts_0",
    "hwsd2_org_carbon",
    "hwsd2_total_n",
    "hwsd2_cn_ratio",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def ensure_new_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output path already exists: {path}")
    path.mkdir(parents=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_uint32(value: str, seed: int = 0) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                json_safe(payload),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def git_info(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, encoding="utf-8"
        ).strip()

    status = run("status", "--porcelain")
    return {
        "repo": str(repo.resolve()),
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def environment_info() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for module_name in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "torch"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - audit path
            versions[module_name] = f"unavailable:{type(exc).__name__}"
    return {
        "created_at": now_iso(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def point_to_spatial_block(point_id: str, resolution_100km: bool = True) -> str:
    value = str(point_id).strip()
    if not value.isdigit() or len(value) < 8:
        return f"HASH_{stable_uint32(value) % 10000:04d}"
    numeric = int(value)
    east = numeric // 1_000_000
    north = (numeric % 10_000) // 100
    if not resolution_100km:
        east = numeric // 100_000
        north = (numeric % 100_000) // 10
    return f"E{east:03d}_N{north:03d}"


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_t, mean_p = float(y_true.mean()), float(y_pred.mean())
    var_t = float(np.var(y_true, ddof=1))
    var_p = float(np.var(y_pred, ddof=1))
    cov = float(np.cov(y_true, y_pred, ddof=1)[0, 1])
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    return float(2.0 * cov / denom) if denom > 0 else float("nan")


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    adjusted_p: int | None = None,
) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) < 2:
        raise ValueError("at least two finite observations are required")
    residual = y_pred - y_true
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    sd = float(np.std(y_true, ddof=1))
    iqr = float(np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))
    adjusted = None
    if adjusted_p is not None and len(y_true) > adjusted_p + 1:
        adjusted = float(
            1.0 - (1.0 - r2) * (len(y_true) - 1) / (len(y_true) - adjusted_p - 1)
        )
    return {
        "n": int(len(y_true)),
        "r2": r2,
        "adjusted_r2": adjusted,
        "rmse": rmse,
        "mae": mae,
        "bias_pred_minus_obs": bias,
        "ccc": ccc(y_true, y_pred),
        "rpd": float(sd / rmse) if rmse > 0 else None,
        "rpiq": float(iqr / rmse) if rmse > 0 else None,
        # No unique, field-standard RPB definition was found. The contract
        # therefore freezes it as NA instead of selecting a favorable formula.
        "rpb": None,
    }


def file_record(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": sha256_file(path) if hash_file else None,
    }


def dataframe_schema_rows(name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "asset": name,
            "column": str(column),
            "dtype": str(frame[column].dtype),
            "missing_n": int(frame[column].isna().sum()),
            "missing_fraction": float(frame[column].isna().mean()),
        }
        for column in frame.columns
    ]


def validate_predictor_names(columns: Iterable[str]) -> None:
    lowered = {str(column).lower() for column in columns}
    bad = sorted(lowered & FORBIDDEN_PREDICTORS)
    bad.extend(
        sorted(
            c for c in lowered if "org_carbon" in c or "total_n" in c or "cn_ratio" in c
        )
    )
    if bad:
        raise ValueError(f"forbidden predictor columns detected: {sorted(set(bad))}")
