from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from d2c.common import (
    atomic_json,
    environment_info,
    git_info,
    json_safe,
    now_iso,
    sha256_file,
)
from d2c_v2.acg_mondrian import apply_mondrian_rcp_absres, apply_mondrian_split_cp
from d2c_v2.common import stable_uint64
from d2c_v2.data_stages import _balanced_fold_map, _block_id
from d2c_v2.modeling import (
    conformal_quantile,
    make_pca_features,
    point_model,
)
from d2c_v2.rcp_absres import apply_rcp_absres, split_cp_interval
from d2c_v2.uq_methods import fit_tau_model, group_labels

STAGES = [f"MDR{index}" for index in range(11)]
STAGE_FOLDERS = {
    "MDR0": "MDR0_governance_and_input_freeze",
    "MDR1": "MDR1_lucas2015_evidence_import",
    "MDR2": "MDR2_lucas2015_incident_boundary",
    "MDR3": "MDR3_lucas2009_all_land_native_layer",
    "MDR4": "MDR4_dataset_roles_and_features",
    "MDR5": "MDR5_implementation_invariants",
    "MDR6": "MDR6_deterministic_smoke",
    "MDR7": "MDR7_lucas2009_formal_matrix",
    "MDR8": "MDR8_austrian_formal_matrix",
    "MDR9": "MDR9_statistics_and_robustness",
    "MDR10": "MDR10_replication_decision",
}

SPLIT = "Split CP"
RCP = "RCP-AbsRes"
MONDRIAN_SPLIT = "Mondrian Split CP"
MONDRIAN_RCP = "Mondrian RCP-AbsRes"
METHODS = [SPLIT, RCP, MONDRIAN_SPLIT, MONDRIAN_RCP]


@dataclass(frozen=True)
class ReplicationData:
    dataset: str
    frame: pd.DataFrame
    features: np.ndarray
    target: np.ndarray
    unit_ids: np.ndarray
    conditional_labels: np.ndarray


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if bool(config["experiment"].get("automatic_retry")):
        raise ValueError("automatic retry must remain disabled")
    if bool(config["claims"].get("trained_model_transfer")):
        raise ValueError(
            "this runner is for independent replication, not trained-model transfer"
        )
    required = [
        Path(config["paths"][name])
        for name in [
            "active_plan",
            "lucas2015_native",
            "lucas2015_predictions",
            "lucas2015_metrics",
            "lucas2015_summary",
            "lucas2015_incident_audit",
            "lucas2009_raw_spectra",
            "lucas2009_audit",
            "austrian_native",
        ]
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing registered input: {missing}")
    return config


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"])


def stage_dir(config: dict[str, Any], stage: str) -> Path:
    return output_root(config) / STAGE_FOLDERS[stage]


def begin_stage(config: dict[str, Any], stage: str) -> tuple[Path, bool]:
    index = STAGES.index(stage)
    if index:
        prior = stage_dir(config, STAGES[index - 1]) / "SUCCESS.json"
        if not prior.exists():
            raise RuntimeError(f"prior stage is incomplete: {STAGES[index - 1]}")
    out = stage_dir(config, stage)
    if (out / "SUCCESS.json").exists():
        return out, True
    if out.exists():
        raise RuntimeError(f"incomplete attempt requires manual review: {out}")
    out.mkdir(parents=True, exist_ok=False)
    return out, False


def _file_manifest(root: Path, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    omit = set(excluded)
    rows: list[dict[str, Any]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name in omit or path.name.endswith(".part"):
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def finish_stage(
    config: dict[str, Any],
    config_path: Path,
    repo: Path,
    out: Path,
    stage: str,
    started: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "run_id": config["experiment"]["run_id"],
        "stage": stage,
        "created_at": now_iso(),
        "elapsed_seconds": float(time.time() - started),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "active_plan_sha256": sha256_file(Path(config["paths"]["active_plan"])),
        "environment": environment_info(),
        "git": git_info(repo),
        "automatic_retry_performed": False,
        "files": _file_manifest(out, {"run_manifest.json", "SUCCESS.json"}),
    }
    atomic_json(out / "run_manifest.json", manifest)
    result = {
        "status": "COMPLETED",
        "created_at": now_iso(),
        "stage": stage,
        **json_safe(payload),
        "automatic_retry_performed": False,
    }
    atomic_json(out / "SUCCESS.json", result)
    return result


def _hash_array(array: np.ndarray) -> str:
    view = np.asarray(array)
    return hashlib.sha256(view.tobytes(order="C")).hexdigest()


def _read_zarr_v2_uncompressed(path: Path) -> np.ndarray:
    metadata = json.loads((path / ".zarray").read_text(encoding="utf-8"))
    if metadata.get("compressor") is not None or metadata.get("filters"):
        raise ValueError(
            "minimal reader supports only uncompressed, unfiltered Zarr v2 arrays"
        )
    shape = tuple(map(int, metadata["shape"]))
    chunks = tuple(map(int, metadata["chunks"]))
    dtype = np.dtype(metadata["dtype"])
    order = str(metadata.get("order", "C"))
    fill = metadata.get("fill_value", None)
    output = np.empty(shape, dtype=dtype)
    for row_chunk in range(math.ceil(shape[0] / chunks[0])):
        for col_chunk in range(math.ceil(shape[1] / chunks[1])):
            chunk_path = path / f"{row_chunk}.{col_chunk}"
            block = np.full(chunks, np.nan if fill is None else fill, dtype=dtype)
            if chunk_path.exists():
                raw = np.fromfile(chunk_path, dtype=dtype)
                expected = int(np.prod(chunks))
                if raw.size != expected:
                    raise ValueError(f"unexpected Zarr chunk size: {chunk_path}")
                block = raw.reshape(chunks, order=order)
            row_start, col_start = row_chunk * chunks[0], col_chunk * chunks[1]
            row_stop = min(row_start + chunks[0], shape[0])
            col_stop = min(col_start + chunks[1], shape[1])
            output[row_start:row_stop, col_start:col_stop] = block[
                : row_stop - row_start, : col_stop - col_start
            ]
    return output


def _physical_sg_window(wavelength: np.ndarray, physical_width_nm: float) -> int:
    step = float(np.median(np.diff(np.asarray(wavelength, dtype=float))))
    points = max(5, int(round(float(physical_width_nm) / step)) + 1)
    if points % 2 == 0:
        points += 1
    return points


def preprocess_to_grid(
    native: np.ndarray,
    wavelength: np.ndarray,
    *,
    start_nm: float,
    end_nm: float,
    step_nm: float,
    physical_window_nm: float,
    polynomial_order: int,
    epsilon: float,
    output_path: Path | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    x = np.asarray(native)
    wave = np.asarray(wavelength, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(wave):
        raise ValueError("native spectra and wavelength grid are unaligned")
    if not np.isfinite(x).all() or not np.all(np.diff(wave) > 0):
        raise ValueError("native spectra or wavelength grid is invalid")
    target_wave = np.arange(float(start_nm), float(end_nm) + 0.1, float(step_nm))
    native_step = float(np.median(np.diff(wave)))
    indices = np.rint((target_wave - wave[0]) / native_step).astype(int)
    if np.max(np.abs(wave[indices] - target_wave)) > 1.0e-6:
        raise ValueError(
            "requested model grid is not represented exactly in native wavelengths"
        )
    shape = (len(x), len(indices))
    if output_path is None:
        output: np.ndarray = np.empty(shape, dtype=np.float32)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=shape
        )
    window = _physical_sg_window(wave, physical_window_nm)
    for start in range(0, len(x), int(batch_size)):
        stop = min(start + int(batch_size), len(x))
        derivative = savgol_filter(
            np.asarray(x[start:stop], dtype=np.float32),
            window_length=window,
            polyorder=int(polynomial_order),
            deriv=1,
            delta=native_step,
            axis=1,
            mode="interp",
        )[:, indices]
        mean = derivative.mean(axis=1, keepdims=True)
        std = derivative.std(axis=1, keepdims=True)
        output[start:stop] = (
            (derivative - mean) / np.maximum(std, float(epsilon))
        ).astype(np.float32)
    if hasattr(output, "flush"):
        output.flush()
    return output


def build_lucas2009_native(config: dict[str, Any]) -> dict[str, Any]:
    destination = Path(config["paths"]["lucas2009_native"])
    validation_path = destination / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if int(validation["sample_n"]) != int(
            config["data"]["lucas2009_triple_intersection_n"]
        ):
            raise RuntimeError(
                "existing LUCAS 2009 native layer has the wrong cohort size"
            )
        return validation
    if destination.exists():
        raise RuntimeError(
            f"incomplete LUCAS 2009 native layer requires manual review: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    audit = pd.read_parquet(Path(config["paths"]["lucas2009_audit"]))
    cohort = audit.loc[audit["main_contract_complete"].astype(bool)].copy()
    cohort["sample_id"] = cohort["sample_id"].astype(str)
    cohort = cohort.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    expected = int(config["data"]["lucas2009_triple_intersection_n"])
    if len(cohort) != expected or cohort["sample_id"].duplicated().any():
        raise RuntimeError("LUCAS 2009 cohort identity failed")
    raw_path = Path(config["paths"]["lucas2009_raw_spectra"])
    header = pd.read_csv(raw_path, nrows=0)
    spectral_columns = [name for name in header.columns if str(name).startswith("spc.")]
    wavelength = np.asarray(
        [float(str(name).split("spc.", 1)[1]) for name in spectral_columns]
    )
    if (
        len(spectral_columns) != 4200
        or wavelength[0] != 400.0
        or wavelength[-1] != 2499.5
    ):
        raise RuntimeError("unexpected LUCAS 2009 native wavelength grid")
    matrix_path = destination / "spectra_mean_native.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(len(cohort), len(wavelength))
    )
    matrix[:] = np.nan
    row_of = {value: index for index, value in enumerate(cohort["sample_id"])}
    counts = np.zeros(len(cohort), dtype=np.int16)
    scan_rows: list[dict[str, Any]] = []
    source_row = 0
    usecols = ["sample.ID", "ID", *spectral_columns]
    for chunk in pd.read_csv(
        raw_path,
        usecols=usecols,
        dtype={"sample.ID": "string", "ID": "string"},
        chunksize=256,
        low_memory=False,
    ):
        sample_ids = chunk["sample.ID"].astype(str).to_numpy()
        spectra = chunk[spectral_columns].to_numpy(dtype=np.float32, copy=True)
        for local_row, sample_id in enumerate(sample_ids):
            target_row = row_of.get(sample_id)
            if target_row is None:
                continue
            if not np.isfinite(spectra[local_row]).all():
                raise RuntimeError(f"non-finite LUCAS 2009 spectrum: {sample_id}")
            if counts[target_row] == 0:
                matrix[target_row] = spectra[local_row]
            else:
                matrix[target_row] += spectra[local_row]
            counts[target_row] += 1
            scan_rows.append(
                {
                    "scan_row": len(scan_rows),
                    "sample_id": sample_id,
                    "source_row": source_row + local_row,
                    "source_id": str(chunk.iloc[local_row]["ID"]),
                }
            )
        source_row += len(chunk)
    if np.any(counts == 0):
        raise RuntimeError(
            f"missing LUCAS 2009 spectra for {int(np.sum(counts == 0))} samples"
        )
    duplicate = counts > 1
    matrix[duplicate] /= counts[duplicate, None]
    matrix.flush()
    expected_counts = cohort["spectral_scan_count"].to_numpy(dtype=int)
    if not np.array_equal(counts.astype(int), expected_counts):
        raise RuntimeError("LUCAS 2009 scan multiplicity differs from the frozen audit")
    if not np.isfinite(np.load(matrix_path, mmap_mode="r")).all():
        raise RuntimeError("LUCAS 2009 native mean matrix is non-finite")
    cohort["sample_row"] = np.arange(len(cohort), dtype=int)
    cohort["point_id"] = cohort["attr_point_id"].astype(str)
    cohort["country"] = cohort["country"].astype(str)
    cohort["lc1_official"] = cohort["land_cover"].astype(str)
    cohort["lu1_official"] = cohort["land_use"].astype(str)
    cohort["scan_count"] = counts.astype(int)
    keep = [
        "sample_id",
        "point_id",
        "sample_row",
        "scan_count",
        "soc_g_kg",
        "tn_g_kg",
        "longitude",
        "latitude",
        "country",
        "lc1_official",
        "lu1_official",
        "land_cover_parent",
        "land_cover_group",
    ]
    cohort[keep].to_parquet(destination / "sample_index.parquet", index=False)
    cohort[["sample_id", "point_id", "soc_g_kg", "tn_g_kg"]].to_parquet(
        destination / "target_table.parquet", index=False
    )
    pd.DataFrame(scan_rows).to_parquet(destination / "scan_index.parquet", index=False)
    np.save(destination / "wavelength_nm.npy", wavelength)
    validation = {
        "dataset": "LUCAS 2009 all eligible land-cover cohort",
        "sample_n": int(len(cohort)),
        "scan_n": int(counts.sum()),
        "duplicate_sample_n": int(duplicate.sum()),
        "wavelength_n": int(len(wavelength)),
        "wavelength_start_nm": float(wavelength[0]),
        "wavelength_end_nm": float(wavelength[-1]),
        "finite_soc_n": int(np.isfinite(cohort["soc_g_kg"]).sum()),
        "finite_tn_n": int(np.isfinite(cohort["tn_g_kg"]).sum()),
        "all_land_cover_categories_retained": True,
        "raw_source_sha256": sha256_file(raw_path),
    }
    atomic_json(validation_path, validation)
    atomic_json(
        destination / "schema.json",
        {
            "matrix": "spectra_mean_native.npy",
            "row_identity": "sample_index.parquet:sample_row",
            "spectral_dtype": "float32",
            "wavelength_unit": "nm",
            "immutable_raw_source": str(raw_path),
        },
    )
    return validation


def _fold_assignments(
    unit_ids: np.ndarray, seeds: list[int], folds: int
) -> pd.DataFrame:
    units = pd.Series(np.asarray(unit_ids, dtype=str))
    counts = units.value_counts()
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        mapping = _balanced_fold_map(counts, int(folds), int(seed))
        rows.append(
            pd.DataFrame(
                {
                    "sample_row": np.arange(len(units), dtype=int),
                    "unit_id": units.to_numpy(),
                    "seed": int(seed),
                    "outer_fold": units.map(mapping).to_numpy(dtype=int),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _dataset_feature_paths(config: dict[str, Any], dataset: str) -> tuple[Path, Path]:
    out = stage_dir(config, "MDR4")
    return out / f"{dataset}_features.npy", out / f"{dataset}_roles.parquet"


def prepare_dataset(
    config: dict[str, Any], dataset: str, *, create_features: bool
) -> ReplicationData:
    prep = config["preprocessing"]
    feature_path, role_path = _dataset_feature_paths(config, dataset)
    if dataset == "lucas2009":
        root = Path(config["paths"]["lucas2009_native"])
        frame = pd.read_parquet(root / "sample_index.parquet").sort_values("sample_row")
        native = np.load(root / "spectra_mean_native.npy", mmap_mode="r")
        wavelength = np.load(root / "wavelength_nm.npy")
        target = frame["soc_g_kg"].to_numpy(dtype=float)
        unit_ids = (
            frame["point_id"]
            .map(
                lambda value: _block_id(
                    str(value), int(config["data"]["primary_block_km"])
                )
            )
            .to_numpy(dtype=str)
        )
        conditional = frame["lc1_official"].astype(str).str[:1].to_numpy()
        start_nm, end_nm = prep["lucas_start_nm"], prep["lucas_end_nm"]
    elif dataset == "austrian":
        root = Path(config["paths"]["austrian_native"])
        sample = pd.read_parquet(root / "sample_index.parquet")
        target_table = pd.read_parquet(root / "target_table.parquet")
        frame = sample.merge(
            target_table[["sample_id", "soc_g_kg", "experiment_number"]],
            on="sample_id",
            how="left",
            suffixes=("", "_target"),
            validate="one_to_one",
        )
        if "soc_g_kg_target" in frame:
            frame["soc_g_kg"] = frame["soc_g_kg_target"]
        eligible = np.isfinite(frame["soc_g_kg"].to_numpy(dtype=float))
        frame = frame.loc[eligible].reset_index(drop=True)
        frame["sample_row"] = np.arange(len(frame), dtype=int)
        native_all = _read_zarr_v2_uncompressed(root / "spectra_native.zarr")
        native = native_all[eligible]
        wavelength = np.load(root / "wavelength_nm.npy")
        target = frame["soc_g_kg"].to_numpy(dtype=float)
        unit_ids = frame["experiment_number"].astype(str).to_numpy()
        conditional = frame["land_use"].fillna("unknown").astype(str).to_numpy()
        start_nm, end_nm = prep["austrian_start_nm"], prep["austrian_end_nm"]
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    if not np.isfinite(target).all() or len(np.unique(unit_ids)) < 5:
        raise RuntimeError(f"invalid target or grouping for {dataset}")
    if create_features:
        features = preprocess_to_grid(
            native,
            wavelength,
            start_nm=float(start_nm),
            end_nm=float(end_nm),
            step_nm=float(prep["model_grid_step_nm"]),
            physical_window_nm=float(prep["sg_physical_window_nm"]),
            polynomial_order=int(prep["sg_polynomial_order"]),
            epsilon=float(prep["snv_epsilon"]),
            output_path=feature_path,
        )
        roles = _fold_assignments(
            unit_ids,
            [int(value) for value in config["experiment"]["seeds"]],
            int(config["experiment"]["outer_folds"]),
        )
        roles.to_parquet(role_path, index=False)
    else:
        features = np.load(feature_path, mmap_mode="r")
    if len(features) != len(frame):
        raise RuntimeError(f"feature/sample alignment failed for {dataset}")
    frame["unit_id"] = unit_ids
    frame["conditional_label"] = conditional
    return ReplicationData(dataset, frame, features, target, unit_ids, conditional)


def _role_indices(assignments: np.ndarray, outer_fold: int) -> dict[str, np.ndarray]:
    role_fold = {
        "D_eval": int(outer_fold),
        "D_cal": (int(outer_fold) + 1) % 5,
        "D_fit": (int(outer_fold) + 2) % 5,
        "D_model": ((int(outer_fold) + 3) % 5, (int(outer_fold) + 4) % 5),
    }
    output = {
        "D_eval": np.flatnonzero(assignments == role_fold["D_eval"]),
        "D_cal": np.flatnonzero(assignments == role_fold["D_cal"]),
        "D_fit": np.flatnonzero(assignments == role_fold["D_fit"]),
        "D_model": np.flatnonzero(np.isin(assignments, role_fold["D_model"])),
    }
    if min(map(len, output.values())) < 20:
        raise RuntimeError(
            "outer role rotation produced a role with fewer than 20 samples"
        )
    names = list(output)
    if any(
        np.intersect1d(output[names[i]], output[names[j]]).size
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ):
        raise RuntimeError("outer roles overlap")
    return output


def _calibration_subset(
    indices: np.ndarray,
    unit_ids: np.ndarray,
    fraction: float,
    seed: int,
    fold: int,
) -> np.ndarray:
    if float(fraction) >= 1.0:
        return np.asarray(indices, dtype=int)
    local_units = np.asarray(unit_ids, dtype=str)[indices]
    unique = sorted(
        np.unique(local_units),
        key=lambda value: stable_uint64(str(value), int(seed + fold * 10007)),
    )
    keep_n = max(1, int(math.ceil(len(unique) * float(fraction))))
    return np.asarray(indices, dtype=int)[np.isin(local_units, unique[:keep_n])]


def _minimum_group(config: dict[str, Any], groups: np.ndarray) -> int:
    if config["rcp"].get("fixed_minimum_group", False):
        return int(config["rcp"]["minimum_calibration_group_n"])
    counts = [
        int(np.sum(np.asarray(groups, dtype=str) == value))
        for value in np.unique(groups)
    ]
    return min(
        int(config["rcp"]["minimum_calibration_group_n"]),
        max(10, int(math.floor(min(counts) * 0.8))),
    )


def _tau_config(config: dict[str, Any]) -> dict[str, Any]:
    source = config["rcp"]
    return {
        "tau_max_iter": int(source["tau_max_iter"]),
        "tau_max_leaf_nodes": int(source["tau_max_leaf_nodes"]),
        "tau_min_samples_leaf": int(source["tau_min_samples_leaf"]),
        "tau_learning_rate": float(source["tau_learning_rate"]),
        "tau_l2_regularization": float(source["tau_l2_regularization"]),
    }


def _scenario_id(nominal: float, high_quantile: float, fraction: float) -> str:
    coverage = int(round(nominal * 100))
    threshold = int(round(high_quantile * 100))
    calibration = int(round(fraction * 100))
    return f"cov{coverage:02d}_high{threshold:02d}_cal{calibration:03d}"


def scenarios(config: dict[str, Any]) -> list[tuple[float, float, float]]:
    primary = float(config["experiment"]["primary_coverage"])
    high = float(config["experiment"]["high_quantile"])
    values = [(primary, high, 1.0)]
    values.extend(
        (float(nominal), high, 1.0)
        for nominal in config["experiment"]["nominal_coverages"]
    )
    values.extend(
        (primary, float(value), 1.0)
        for value in config["experiment"]["high_quantile_sensitivity"]
    )
    values.extend(
        (primary, high, float(value))
        for value in config["experiment"]["calibration_fractions"]
    )
    return list(dict.fromkeys(values))


def _point_and_context(
    data: ReplicationData,
    roles: dict[str, np.ndarray],
    *,
    seed: int,
    config: dict[str, Any],
    frozen_point: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model_cfg = config["point_model"]
    model = point_model(
        "ExtraTrees", None, int(seed), extra_trees_n_jobs=int(model_cfg["n_jobs"])
    )
    model.set_params(
        n_estimators=int(model_cfg["estimators"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        max_features=float(model_cfg["max_features"]),
    )
    if frozen_point is None:
        model.fit(data.features[roles["D_model"]], data.target[roles["D_model"]])
        point = {
            name: np.asarray(model.predict(data.features[index]), dtype=float)
            for name, index in roles.items()
            if name != "D_model"
        }
    else:
        point = {
            name: np.asarray(frozen_point[name], dtype=float).copy()
            for name in ("D_fit", "D_cal", "D_eval")
        }
        for name, values in point.items():
            if values.shape != (len(roles[name]),) or not np.isfinite(values).all():
                raise ValueError(f"invalid frozen point predictions: {name}")
    _, transformed, _ = make_pca_features(
        data.features[roles["D_model"]],
        [data.features[roles[name]] for name in ["D_fit", "D_cal", "D_eval"]],
        components=int(config["rcp"]["pca_components"]),
        seed=int(seed),
    )
    context = {
        name: np.column_stack([value, point[name]]).astype(np.float32)
        for name, value in zip(["D_fit", "D_cal", "D_eval"], transformed)
    }
    return point, context


def evaluate_outer_cell(
    data: ReplicationData,
    assignments: np.ndarray,
    *,
    seed: int,
    fold: int,
    config: dict[str, Any],
    audit_path: Path | None = None,
    frozen_point: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    roles = _role_indices(assignments, int(fold))
    for name, index in roles.items():
        role_units = set(np.asarray(data.unit_ids, dtype=str)[index])
        for other_name, other_index in roles.items():
            if name >= other_name:
                continue
            if role_units & set(np.asarray(data.unit_ids, dtype=str)[other_index]):
                raise RuntimeError(f"group leakage between {name} and {other_name}")
    point, context = _point_and_context(
        data,
        roles,
        seed=int(seed + fold * 1000),
        config=config,
        frozen_point=frozen_point,
    )
    fit_index, cal_index, eval_index = roles["D_fit"], roles["D_cal"], roles["D_eval"]
    observed_fit = data.target[fit_index]
    observed_eval = data.target[eval_index]
    abs_fit = np.abs(observed_fit - point["D_fit"])
    q0_by_nominal: dict[float, float] = {}
    tau_by_nominal: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    nominal_values = sorted({value[0] for value in scenarios(config)})
    for nominal in nominal_values:
        tau_cal, tau_eval = fit_tau_model(
            context["D_fit"],
            abs_fit,
            [context["D_cal"], context["D_eval"]],
            nominal=float(nominal),
            config=_tau_config(config),
            seed=int(seed + fold * 1000 + round(nominal * 100) + 11),
        )
        tau_by_nominal[float(nominal)] = (tau_cal, tau_eval)
        q0_by_nominal[float(nominal)] = conformal_quantile(
            abs_fit, 1.0 - float(nominal)
        )
    if audit_path is not None:
        audit_path.mkdir(parents=True, exist_ok=False)
        arrays = {f"index_{name}": index for name, index in roles.items()}
        arrays.update({f"point_{name}": value for name, value in point.items()})
        arrays.update(
            {f"observed_{name}": data.target[index] for name, index in roles.items()}
        )
        for nominal, (tau_cal, tau_eval) in tau_by_nominal.items():
            arrays[f"tau_cal_{nominal:.2f}"] = tau_cal
            arrays[f"tau_eval_{nominal:.2f}"] = tau_eval
        np.savez_compressed(audit_path / "calibration_arrays.npz", **arrays)
    rows: list[pd.DataFrame] = []
    for nominal, high_quantile, fraction in scenarios(config):
        used_cal_index = _calibration_subset(
            cal_index, data.unit_ids, fraction, int(seed), int(fold)
        )
        cal_positions = pd.Index(cal_index).get_indexer(used_cal_index)
        if np.any(cal_positions < 0):
            raise RuntimeError("calibration subset indexing failed")
        observed_cal = data.target[used_cal_index]
        point_cal = point["D_cal"][cal_positions]
        point_eval = point["D_eval"]
        threshold = float(np.quantile(point["D_fit"], float(high_quantile)))
        group_cal = group_labels(point_cal, threshold)
        group_eval = group_labels(point_eval, threshold)
        minimum_group = _minimum_group(config, group_cal)
        tau_cal_all, tau_eval = tau_by_nominal[float(nominal)]
        tau_cal = tau_cal_all[cal_positions]
        split_lower, split_upper, split_q = split_cp_interval(
            nominal=nominal,
            observed_cal=observed_cal,
            point_cal=point_cal,
            point_eval=point_eval,
        )
        rcp_lower, rcp_upper, rcp_meta = apply_rcp_absres(
            nominal=nominal,
            observed_cal=observed_cal,
            point_cal=point_cal,
            point_eval=point_eval,
            tau_cal=tau_cal,
            tau_eval=tau_eval,
            q0=q0_by_nominal[float(nominal)],
            selected_lambda=float(config["rcp"]["lambda"]),
        )
        mondrian_split_lower, mondrian_split_upper, mondrian_split_meta = (
            apply_mondrian_split_cp(
                nominal=nominal,
                observed_cal=observed_cal,
                point_cal=point_cal,
                point_eval=point_eval,
                group_cal=group_cal,
                group_eval=group_eval,
                minimum_group_n=minimum_group,
            )
        )
        mondrian_rcp_lower, mondrian_rcp_upper, mondrian_rcp_meta = (
            apply_mondrian_rcp_absres(
                nominal=nominal,
                observed_cal=observed_cal,
                point_cal=point_cal,
                point_eval=point_eval,
                tau_cal=tau_cal,
                tau_eval=tau_eval,
                q0=q0_by_nominal[float(nominal)],
                group_cal=group_cal,
                group_eval=group_eval,
                minimum_group_n=minimum_group,
                selected_lambda=float(config["rcp"]["lambda"]),
            )
        )
        method_values = [
            (SPLIT, split_lower, split_upper, {"q_global": split_q}),
            (RCP, rcp_lower, rcp_upper, rcp_meta),
            (
                MONDRIAN_SPLIT,
                mondrian_split_lower,
                mondrian_split_upper,
                mondrian_split_meta,
            ),
            (MONDRIAN_RCP, mondrian_rcp_lower, mondrian_rcp_upper, mondrian_rcp_meta),
        ]
        template = data.frame.iloc[eval_index].reset_index(drop=True)
        scenario_name = _scenario_id(nominal, high_quantile, fraction)
        for method, lower, upper, metadata in method_values:
            result = pd.DataFrame(
                {
                    "dataset": data.dataset,
                    "target": str(config["experiment"].get("primary_target", "SOC")),
                    "method": method,
                    "seed": int(seed),
                    "outer_fold": int(fold),
                    "scenario": scenario_name,
                    "nominal_coverage": float(nominal),
                    "high_quantile": float(high_quantile),
                    "calibration_fraction": float(fraction),
                    "sample_row": template["sample_row"].to_numpy(dtype=int),
                    "sample_id": template["sample_id"].astype(str).to_numpy(),
                    "unit_id": template["unit_id"].astype(str).to_numpy(),
                    "conditional_label": template["conditional_label"]
                    .astype(str)
                    .to_numpy(),
                    "observed": observed_eval,
                    "predicted": point_eval,
                    "lower": np.asarray(lower, dtype=float),
                    "upper": np.asarray(upper, dtype=float),
                    "group_high_rest": group_eval,
                    "difficulty_tau": np.asarray(tau_eval, dtype=float),
                    "calibration_metadata": json.dumps(
                        json_safe(metadata), sort_keys=True
                    ),
                }
            )
            rows.append(result)
    output = pd.concat(rows, ignore_index=True)
    identity = output.groupby(["scenario", "sample_row"]).agg(
        observed_range=("observed", lambda value: float(value.max() - value.min())),
        predicted_range=("predicted", lambda value: float(value.max() - value.min())),
        method_n=("method", "nunique"),
    )
    if identity[["observed_range", "predicted_range"]].to_numpy().max() > 1.0e-12:
        raise RuntimeError("matched endpoint identity failed")
    if int(identity["method_n"].min()) != len(METHODS):
        raise RuntimeError("four-method endpoint matrix is incomplete")
    status = {
        "dataset": data.dataset,
        "seed": int(seed),
        "outer_fold": int(fold),
        "scenario_n": int(output["scenario"].nunique()),
        "method_n": int(output["method"].nunique()),
        "evaluation_n": int(len(eval_index)),
        "elapsed_seconds": float(time.time() - started),
        "target": str(config["experiment"].get("primary_target", "SOC")),
        "role_n": {name: len(index) for name, index in roles.items()},
        "training_target_mean": float(np.mean(data.target[roles["D_model"]])),
    }
    return output, status
