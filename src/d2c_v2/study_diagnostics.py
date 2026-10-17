"""Aggregate residual diagnostics and recalibration across grouping thresholds.

The functions consume aligned arrays from one fitted cell. They neither fit
models nor draw figures, and their outputs contain no sample-level records.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from d2c_v2.modeling import interval_score
from d2c_v2.protocol_utils import full_cohort_intervals, quantile

THRESHOLDS = tuple(percent / 100 for percent in range(50, 91))


def _vector(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def residual_diagnostics(
    point_fit,
    observed_cal,
    point_cal,
    tau_cal,
    units_cal,
    observed_eval,
    point_eval,
    tau_eval,
    units_eval,
) -> pd.DataFrame:
    """Describe calibration/evaluation residuals on fixed D_fit prediction bins.

    Empirical quantiles use linear interpolation. The global conformal radius
    instead uses the finite-sample order statistic. Centering is performed within
    each group and partition, not with the complete evaluation-set median.
    """
    fit = _vector(point_fit, "point_fit")
    y_cal = _vector(observed_cal, "observed_cal")
    p_cal = _vector(point_cal, "point_cal")
    if y_cal.shape != p_cal.shape:
        raise ValueError("unaligned calibration observations and predictions")
    radius = quantile(np.abs(y_cal - p_cal))
    boundaries = np.quantile(fit, np.arange(1, 10) / 10, method="linear")
    records = []
    for partition, y, point, tau, units in (
        ("calibration", y_cal, p_cal, tau_cal, units_cal),
        ("evaluation", observed_eval, point_eval, tau_eval, units_eval),
    ):
        y = _vector(y, "observed")
        point = _vector(point, "point")
        tau = _vector(tau, "tau")
        units = np.asarray(units)
        if not y.shape == point.shape == tau.shape == units.shape:
            raise ValueError("unaligned diagnostic arrays")
        if pd.isna(units).any():
            raise ValueError("missing evaluation-unit identifier")
        bins = np.searchsorted(boundaries, point, side="right")
        high = point >= boundaries[7]
        groups = [
            ("overall", np.ones(len(y), dtype=bool)),
            ("high", high),
            ("rest", ~high),
        ]
        groups += [(f"decile_{i + 1}", bins == i) for i in range(10)]
        residual = y - point
        for group, take in groups:
            if not take.any():
                raise ValueError(f"empty {partition}/{group}; no diagnostic imputation")
            e = residual[take]
            absolute = np.abs(e)
            q10, q25, median, q75, q90 = np.quantile(
                e, [0.1, 0.25, 0.5, 0.75, 0.9], method="linear"
            )
            covered = (y[take] >= point[take] - radius) & (
                y[take] <= point[take] + radius
            )
            lower_miss = y[take] < point[take] - radius
            upper_miss = y[take] > point[take] + radius
            records.append(
                dict(
                    partition=partition,
                    group=group,
                    n=int(take.sum()),
                    units=int(pd.Series(units[take]).nunique()),
                    signed_mean=float(e.mean()),
                    signed_median=float(median),
                    signed_q10=float(q10),
                    signed_q90=float(q90),
                    signed_iqr=float(q75 - q25),
                    signed_mad=float(np.quantile(np.abs(e - median), 0.5)),
                    raw_q90=float(np.quantile(absolute, 0.9)),
                    median_centered_q90=float(np.quantile(np.abs(e - median), 0.9)),
                    rectified_q90=float(np.quantile(absolute - tau[take], 0.9)),
                    global_radius=radius,
                    coverage=float(covered.mean()),
                    lower_miss=float(lower_miss.mean()),
                    upper_miss=float(upper_miss.mean()),
                    endpoint_vs_score_disagreements=int(
                        np.count_nonzero(covered != (absolute <= radius))
                    ),
                )
            )
    return pd.DataFrame(records)


def tail_gaps(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Calculate high-minus-rest contrasts within a cell, before summarizing."""
    high = diagnostics.loc[diagnostics.group.eq("high")].set_index("partition")
    rest = diagnostics.loc[diagnostics.group.eq("rest")].set_index("partition")
    if len(high) != 2 or set(high.index) != {"calibration", "evaluation"}:
        raise ValueError("one calibration and one evaluation row per group required")
    if not high.index.equals(rest.index):
        raise ValueError("unaligned high/rest diagnostic rows")
    result = pd.DataFrame(index=high.index)
    for name in ("raw_q90", "rectified_q90", "median_centered_q90"):
        result[name + "_gap"] = high[name] - rest[name]
    return result.reset_index()


def threshold_statistics(
    point_fit,
    observed_cal,
    point_cal,
    tau_cal,
    observed_eval,
    point_eval,
    tau_eval,
    *,
    thresholds=THRESHOLDS,
    minimum_group_size: int = 50,
) -> pd.DataFrame:
    """Recalibrate four fixed predictors at every threshold; never refit models.

    Overall widths/scores use the complete evaluation set. High-group coverage
    uses only the high group for the current threshold. These are separate
    calibrated analyses, not simultaneous coverage guarantees for nested groups.
    """
    fit = _vector(point_fit, "point_fit")
    y_cal, p_cal, t_cal, y_eval, p_eval, t_eval = [
        _vector(v, name)
        for v, name in zip(
            (observed_cal, point_cal, tau_cal, observed_eval, point_eval, tau_eval),
            (
                "observed_cal",
                "point_cal",
                "tau_cal",
                "observed_eval",
                "point_eval",
                "tau_eval",
            ),
        )
    ]
    if (
        not y_cal.shape == p_cal.shape == t_cal.shape
        or not y_eval.shape == p_eval.shape == t_eval.shape
    ):
        raise ValueError("unaligned threshold-analysis arrays")
    requested = tuple(float(v) for v in thresholds)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(not 0 < v < 1 for v in requested)
    ):
        raise ValueError(
            "threshold quantiles must be distinct values between zero and one"
        )
    records = []
    for gamma in requested:
        threshold = float(np.quantile(fit, gamma, method="linear"))
        gc = np.where(p_cal >= threshold, "high", "rest")
        ge = np.where(p_eval >= threshold, "high", "rest")
        intervals, _ = full_cohort_intervals(
            y_cal,
            p_cal,
            p_eval,
            t_cal,
            t_eval,
            gc,
            ge,
            minimum_group_size=minimum_group_size,
        )
        high = ge == "high"
        if not high.any() or high.all():
            raise ValueError("empty evaluation group at requested threshold")
        for method, (lower, upper) in intervals.items():
            covered = (lower <= y_eval) & (y_eval <= upper)
            width = upper - lower
            records.append(
                dict(
                    high_quantile=gamma,
                    method=method,
                    n=int(len(y_eval)),
                    n_high=int(high.sum()),
                    n_rest=int((~high).sum()),
                    calibration_high_n=int((gc == "high").sum()),
                    calibration_rest_n=int((gc == "rest").sum()),
                    coverage=float(covered.mean()),
                    high_coverage=float(covered[high].mean()),
                    rest_coverage=float(covered[~high].mean()),
                    mean_width=float(width.mean()),
                    high_mean_width=float(width[high].mean()),
                    rest_mean_width=float(width[~high].mean()),
                    mean_interval_score=float(
                        interval_score(y_eval, lower, upper, alpha=0.1).mean()
                    ),
                )
            )
    return pd.DataFrame(records)


def write_cell_statistics(
    output: Path,
    *,
    task: str,
    seed: int,
    fold: int,
    point_fit,
    observed_cal,
    point_cal,
    tau_cal,
    units_cal,
    observed_eval,
    point_eval,
    tau_eval,
    units_eval,
    threshold_scan: bool = False,
) -> None:
    """Write compact numerical results, without any sample identifiers/values."""
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = residual_diagnostics(
        point_fit,
        observed_cal,
        point_cal,
        tau_cal,
        units_cal,
        observed_eval,
        point_eval,
        tau_eval,
        units_eval,
    )
    tables = {
        "residual_diagnostics.csv": diagnostics,
        "tail_gaps.csv": tail_gaps(diagnostics),
    }
    # The reference is always calculated for pooled group-width reconstruction.
    widths = threshold_statistics(
        point_fit,
        observed_cal,
        point_cal,
        tau_cal,
        observed_eval,
        point_eval,
        tau_eval,
        thresholds=(0.8,),
    )
    tables["group_widths.csv"] = widths
    if threshold_scan:
        tables["threshold_metrics.csv"] = threshold_statistics(
            point_fit,
            observed_cal,
            point_cal,
            tau_cal,
            observed_eval,
            point_eval,
            tau_eval,
        )
    for name, table in tables.items():
        target = output / name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        table.assign(task=task, seed=seed, fold=fold).to_csv(target, index=False)
