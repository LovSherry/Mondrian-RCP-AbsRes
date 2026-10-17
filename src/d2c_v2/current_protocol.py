"""Portable training entry point for the experiments reported in the study.

Only prepared data and saved fold assignments are inputs. No archived predictions,
fitted models, manuscript files, internal plans, or machine-specific paths are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from d2c_v2.modeling import conformal_quantile, make_pca_features, point_model
from d2c_v2.multi_dataset_replication import (
    ReplicationData,
    _role_indices,
    evaluate_outer_cell,
)
from d2c_v2.protocol_utils import (
    check_roles,
    cqr_interval,
    full_cohort_intervals,
    summaries,
)
from d2c_v2.study_diagnostics import write_cell_statistics

SEEDS = tuple(range(20260831, 20260836))
TASKS = {
    "2015_SOC": ("lucas2015", "SOC", "full_cohort", 21782),
    "2015_TN": ("lucas2015", "TN", "full_cohort", 21782),
    "2009_SOC": ("lucas2009", "SOC", "ablation", 18739),
    "Austria_SOC": ("austrian", "SOC", "ablation", 2112),
    "Clay": ("clay", "Clay", "ablation", 18739),
    "2009_public": ("lucas2009", "SOC", "public", 18739),
    "Austria_public": ("austrian", "SOC", "public", 2112),
}
HGB = dict(
    max_iter=100,
    max_leaf_nodes=15,
    min_samples_leaf=40,
    learning_rate=0.05,
    l2_regularization=2.0,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=10,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task(root: Path, task: str, seed: int):
    dataset, target, family, expected_n = TASKS[task]
    folder = root / dataset
    metadata = json.loads((folder / "dataset.json").read_text(encoding="utf8"))
    paths = [
        folder / name
        for name in ("dataset.json", "features.npy", "samples.parquet", "roles.parquet")
    ]
    hashes = {p.name: file_hash(p) for p in paths}
    if hashes["features.npy"] != metadata["features_sha256"]:
        raise ValueError("prepared spectral checksum mismatch")
    frame = pd.read_parquet(folder / "samples.parquet")
    x = np.load(folder / "features.npy", allow_pickle=False)
    y = frame[metadata["targets"][target]].to_numpy(float)
    roles = pd.read_parquet(folder / "roles.parquet")
    a = roles.loc[roles.seed.eq(seed)].sort_values("sample_row")
    if len(frame) != expected_n or x.shape != (expected_n, metadata["p"]):
        raise ValueError("prepared data do not match the current cohort dimensions")
    if not frame.sample_id.astype(str).is_unique:
        raise ValueError("duplicate prepared sample identifiers")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite prepared input")
    np.testing.assert_array_equal(frame.sample_row.to_numpy(), np.arange(expected_n))
    np.testing.assert_array_equal(a.sample_row.to_numpy(), np.arange(expected_n))
    np.testing.assert_array_equal(a.unit_id.astype(str), frame.unit_id.astype(str))
    if (
        set(a.outer_fold) != set(range(5))
        or a.groupby("unit_id").outer_fold.nunique().max() != 1
    ):
        raise ValueError("invalid or leaking spatial folds")
    return dataset, target, family, frame, x, y, a.outer_fold.to_numpy(int), hashes


def ablation_config(target: str) -> dict:
    return {
        "experiment": dict(
            primary_target=target,
            primary_coverage=0.9,
            high_quantile=0.8,
            nominal_coverages=[0.9],
            high_quantile_sensitivity=[],
            calibration_fractions=[1.0],
        ),
        "point_model": dict(
            estimators=256, min_samples_leaf=3, max_features=0.7, n_jobs=1
        ),
        "rcp": dict(
            pca_components=32,
            tau_max_iter=100,
            tau_max_leaf_nodes=15,
            tau_min_samples_leaf=40,
            tau_learning_rate=0.05,
            tau_l2_regularization=2.0,
            minimum_calibration_group_n=50,
            fixed_minimum_group=True,
            **{"lambda": 1.0},
        ),
    }


def fit_intervals(
    frame,
    x,
    y,
    roles,
    seed,
    fold,
    family,
    target,
    *,
    statistics_output=None,
    task=None,
    threshold_scan=False,
):
    """Full-cohort interval fits and the two independently fitted public baselines."""
    run_seed = seed + fold * 1000
    model = point_model("ExtraTrees", None, run_seed, extra_trees_n_jobs=1)
    model.fit(x[roles["D_model"]], y[roles["D_model"]])
    point = {k: model.predict(x[ix]) for k, ix in roles.items() if k != "D_model"}
    ptrain, arrays, _ = make_pca_features(
        x[roles["D_model"]],
        [x[roles[k]] for k in ("D_fit", "D_cal", "D_eval")],
        components=32,
        seed=run_seed,
    )
    pca = dict(zip(("D_fit", "D_cal", "D_eval"), arrays))
    ctx = {k: np.column_stack([pca[k], point[k]]).astype(np.float32) for k in pca}
    residual = np.abs(y[roles["D_fit"]] - point["D_fit"])
    threshold = float(np.quantile(point["D_fit"], 0.8))
    gc = np.where(point["D_cal"] >= threshold, "high", "rest")
    ge = np.where(point["D_eval"] >= threshold, "high", "rest")
    intervals = {}
    diagnostics = {
        "high_threshold": threshold,
        "calibration_group_n": {g: int((gc == g).sum()) for g in ("high", "rest")},
    }
    if min(diagnostics["calibration_group_n"].values()) < 50:
        raise ValueError(
            "current protocol requires at least 50 calibration samples per group"
        )
    if family == "full_cohort":
        tau = HistGradientBoostingRegressor(
            **HGB, loss="quantile", quantile=0.9, random_state=run_seed + 101
        )
        tau.fit(ctx["D_fit"], residual)
        intervals, diagnostics["calibration"] = full_cohort_intervals(
            y[roles["D_cal"]],
            point["D_cal"],
            point["D_eval"],
            tau.predict(ctx["D_cal"]),
            tau.predict(ctx["D_eval"]),
            gc,
            ge,
        )
        if statistics_output is not None:
            write_cell_statistics(
                statistics_output,
                task=task,
                seed=seed,
                fold=fold,
                point_fit=point["D_fit"],
                observed_cal=y[roles["D_cal"]],
                point_cal=point["D_cal"],
                tau_cal=tau.predict(ctx["D_cal"]),
                units_cal=frame.iloc[roles["D_cal"]].unit_id.to_numpy(),
                observed_eval=y[roles["D_eval"]],
                point_eval=point["D_eval"],
                tau_eval=tau.predict(ctx["D_eval"]),
                units_eval=frame.iloc[roles["D_eval"]].unit_id.to_numpy(),
                threshold_scan=threshold_scan,
            )
    if target == "SOC":
        ncp = HistGradientBoostingRegressor(
            **HGB, loss="squared_error", random_state=run_seed + 601
        )
        ncp.fit(ctx["D_fit"], np.log(residual + 1e-6))
        sc = np.maximum(np.exp(ncp.predict(ctx["D_cal"])), 1e-6)
        se = np.maximum(np.exp(ncp.predict(ctx["D_eval"])), 1e-6)
        qn = conformal_quantile(np.abs(y[roles["D_cal"]] - point["D_cal"]) / sc, 0.1)
        intervals["NCP"] = (point["D_eval"] - qn * se, point["D_eval"] + qn * se)
        train = np.concatenate([roles["D_model"], roles["D_fit"]])
        z = np.concatenate([ptrain, pca["D_fit"]])
        low = HistGradientBoostingRegressor(
            **HGB, loss="quantile", quantile=0.05, random_state=run_seed + 701
        )
        high = HistGradientBoostingRegressor(
            **HGB, loss="quantile", quantile=0.95, random_state=run_seed + 702
        )
        low.fit(z, y[train])
        high.fit(z, y[train])
        lo, hi, raw_q = cqr_interval(
            y[roles["D_cal"]],
            low.predict(pca["D_cal"]),
            high.predict(pca["D_cal"]),
            low.predict(pca["D_eval"]),
            high.predict(pca["D_eval"]),
        )
        intervals["Standard CQR"] = (lo, hi)
        diagnostics.update(ncp_q=qn, cqr_q_raw=raw_q)
    ev = frame.iloc[roles["D_eval"]]
    predictions = []
    for method, (lo, hi) in intervals.items():
        if not np.isfinite(np.column_stack([lo, hi])).all() or (lo > hi).any():
            raise ValueError("invalid prediction interval")
        predictions.append(
            pd.DataFrame(
                dict(
                    method=method,
                    sample_row=ev.sample_row.to_numpy(),
                    sample_id=ev.sample_id.astype(str).to_numpy(),
                    unit_id=ev.unit_id.astype(str).to_numpy(),
                    conditional_label=ev.conditional_label.astype(str).to_numpy(),
                    observed=y[roles["D_eval"]],
                    predicted=point["D_eval"],
                    lower=lo,
                    upper=hi,
                    group_high_rest=ge,
                )
            )
        )
    return pd.concat(predictions, ignore_index=True), diagnostics


def run_cell(
    prepared: Path,
    task: str,
    seed: int,
    fold: int,
    output: Path,
    *,
    threshold_scan: bool = False,
):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    dataset, target, family, frame, x, y, assignments, hashes = load_task(
        prepared, task, seed
    )
    roles = _role_indices(assignments, fold)
    check_roles(frame, roles)
    output.mkdir(parents=True)
    started = time.monotonic()
    if family == "ablation":
        data = ReplicationData(
            "lucas2009_clay" if task == "Clay" else dataset,
            frame,
            x,
            y,
            frame.unit_id.astype(str).to_numpy(),
            frame.conditional_label.astype(str).to_numpy(),
        )
        pred, details = evaluate_outer_cell(
            data,
            assignments,
            seed=seed,
            fold=fold,
            config=ablation_config(target),
            audit_path=output / "audit",
        )
        # The historical implementation is retained; unsupported groups are rejected.
        for encoded in pred.calibration_metadata.unique():
            counts = json.loads(encoded).get("calibration_group_n", {})
            if counts and min(counts.values()) < 50:
                raise ValueError("fewer than 50 calibration samples in a group")
        with np.load(
            output / "audit/calibration_arrays.npz", allow_pickle=False
        ) as arrays:
            write_cell_statistics(
                output / "statistics",
                task=task,
                seed=seed,
                fold=fold,
                point_fit=arrays["point_D_fit"],
                observed_cal=arrays["observed_D_cal"],
                point_cal=arrays["point_D_cal"],
                tau_cal=arrays["tau_cal_0.90"],
                units_cal=frame.iloc[roles["D_cal"]].unit_id.to_numpy(),
                observed_eval=arrays["observed_D_eval"],
                point_eval=arrays["point_D_eval"],
                tau_eval=arrays["tau_eval_0.90"],
                units_eval=frame.iloc[roles["D_eval"]].unit_id.to_numpy(),
                threshold_scan=threshold_scan,
            )
    else:
        pred, details = fit_intervals(
            frame,
            x,
            y,
            roles,
            seed,
            fold,
            family,
            target,
            statistics_output=output / "statistics",
            task=task,
            threshold_scan=threshold_scan,
        )
    pred = pred.assign(
        dataset=dataset,
        target=target,
        seed=seed,
        outer_fold=fold,
        nominal_coverage=0.9,
        high_quantile=0.8,
        calibration_fraction=1.0,
        scenario="cov90_high80_cal100",
    )
    pred.to_parquet(output / "predictions.parquet", index=False)
    summaries(pred, ["method"]).to_csv(output / "metrics.csv", index=False)
    result = dict(
        task=task,
        seed=seed,
        fold=fold,
        seconds=time.monotonic() - started,
        rows=len(pred),
        role_n={k: len(v) for k, v in roles.items()},
        input_sha256=hashes,
        prediction_sha256=file_hash(output / "predictions.parquet"),
        details=details,
        statistics_files={
            p.name: file_hash(p) for p in sorted((output / "statistics").glob("*.csv"))
        },
    )
    (output / "SUCCESS.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=[*TASKS, "all"], default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument(
        "--threshold-scan",
        action="store_true",
        help="also recalculate 50th-90th percentile statistics for LUCAS SOC tasks",
    )
    args = parser.parse_args()
    if args.threshold_scan and args.task not in {"all", "2015_SOC", "2009_SOC"}:
        parser.error("--threshold-scan applies only to the two LUCAS SOC analyses")
    if args.output.exists():
        raise FileExistsError("choose a new output directory")
    args.output.mkdir(parents=True)
    report = dict(
        status="RUNNING",
        study=(
            "Mondrian rectified conformal prediction for conditional reliability "
            "in soil spectroscopic regression"
        ),
        records=[],
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            platform=platform.platform(),
        ),
        training_inputs=(
            "prepared spectra, labels and saved spatial folds; no old predictions "
            "or models"
        ),
        threshold_scan=args.threshold_scan,
    )
    tasks = list(TASKS) if args.task == "all" else [args.task]
    cells = (
        [(s, k) for s in SEEDS for k in range(5)]
        if args.full_matrix
        else [(args.seed, args.fold)]
    )
    with threadpool_limits(limits=1):
        for task in tasks:
            for seed, fold in cells:
                print(f"START {task} {seed} {fold}", flush=True)
                record = run_cell(
                    args.prepared_data,
                    task,
                    seed,
                    fold,
                    args.output / f"{task}_{seed}_{fold}",
                    threshold_scan=args.threshold_scan
                    and task in {"2015_SOC", "2009_SOC"},
                )
                report["records"].append(record)
                (args.output / "RUN_REPORT.json").write_text(
                    json.dumps(report, indent=2), encoding="utf8"
                )
                print(f"DONE {task}: {record['seconds']:.1f}s", flush=True)
    report["status"] = "COMPLETED"
    (args.output / "RUN_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()
