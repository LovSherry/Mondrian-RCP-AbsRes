"""Small, reusable utilities for the public experimental protocol."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from d2c_v2.modeling import conformal_quantile, interval_score

ABLATION_METHODS = (
    "Split CP",
    "RCP-AbsRes",
    "Mondrian Split CP",
    "Mondrian RCP-AbsRes",
)
PUBLIC_METHODS = (
    "Split CP",
    "Standard CQR",
    "NCP",
    "Mondrian RCP-AbsRes",
)


def quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Return the finite-sample conformal order statistic."""
    return conformal_quantile(scores, alpha)


def cqr_interval(
    y_cal: np.ndarray,
    lower_cal: np.ndarray,
    upper_cal: np.ndarray,
    lower_eval: np.ndarray,
    upper_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate conformalized quantile-regression endpoints."""
    low = np.minimum(lower_cal, upper_cal)
    high = np.maximum(lower_cal, upper_cal)
    raw_correction = quantile(np.maximum(low - y_cal, y_cal - high))
    correction = max(raw_correction, 0.0)
    return (
        np.minimum(lower_eval, upper_eval) - correction,
        np.maximum(lower_eval, upper_eval) + correction,
        raw_correction,
    )


def check_roles(frame: pd.DataFrame, roles: dict[str, np.ndarray]) -> None:
    """Verify that sample roles partition the cohort without group leakage."""
    groups = {
        name: set(frame.iloc[index].unit_id.astype(str))
        for name, index in roles.items()
    }
    for first, second in combinations(roles, 2):
        sample_overlap = np.intersect1d(roles[first], roles[second]).size
        group_overlap = groups[first] & groups[second]
        if sample_overlap or group_overlap:
            raise ValueError("role or group leakage")
    assigned = np.concatenate(list(roles.values()))
    if len(np.unique(assigned)) != len(frame):
        raise ValueError("roles do not partition the cohort")


def full_cohort_intervals(
    y_cal: np.ndarray,
    point_cal: np.ndarray,
    point_eval: np.ndarray,
    tau_cal: np.ndarray,
    tau_eval: np.ndarray,
    group_cal: np.ndarray,
    group_eval: np.ndarray,
    *,
    minimum_group_size: int = 50,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Construct the four ablation intervals from shared point predictions."""
    residual = np.abs(y_cal - point_cal)
    split_quantile = quantile(residual)
    rcp_quantile = quantile(residual - tau_cal)
    split_by_group: dict[str, float] = {}
    rcp_by_group: dict[str, float] = {}
    for group in ("high", "rest"):
        take = group_cal == group
        if int(take.sum()) < minimum_group_size:
            raise ValueError(
                f"fewer than {minimum_group_size} calibration samples in {group}"
            )
        split_by_group[group] = quantile(residual[take])
        rcp_by_group[group] = quantile((residual - tau_cal)[take])

    radii = (
        np.full(len(point_eval), split_quantile),
        np.maximum(tau_eval + rcp_quantile, 0.0),
        np.asarray([split_by_group[group] for group in group_eval]),
        np.maximum(
            tau_eval + np.asarray([rcp_by_group[group] for group in group_eval]),
            0.0,
        ),
    )
    intervals = {
        method: (point_eval - radius, point_eval + radius)
        for method, radius in zip(ABLATION_METHODS, radii)
    }
    diagnostics: dict[str, object] = {
        "q": split_quantile,
        "q_rcp": rcp_quantile,
        "q_by_group": split_by_group,
        "q_rcp_by_group": rcp_by_group,
    }
    return intervals, diagnostics


def row_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize point and interval predictions for one analysis cell."""
    y, point, lower, upper = [
        frame[column].to_numpy(float)
        for column in ("observed", "predicted", "lower", "upper")
    ]
    covered = (lower <= y) & (y <= upper)
    high = frame.group_high_rest.eq("high").to_numpy()
    return {
        "n_unique": int(frame.sample_row.nunique()),
        "n_predictions": int(len(frame)),
        "r2": float(1 - np.sum((y - point) ** 2) / np.sum((y - y.mean()) ** 2)),
        "rmse": float(np.sqrt(np.mean((y - point) ** 2))),
        "mae": float(np.mean(np.abs(y - point))),
        "coverage": float(covered.mean()),
        "high_coverage": float(covered[high].mean()),
        "rest_coverage": float(covered[~high].mean()),
        "mean_width": float(np.mean(upper - lower)),
        "mean_interval_score": float(interval_score(y, lower, upper, alpha=0.1).mean()),
    }


def summaries(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Apply :func:`row_metrics` to groups defined by ``keys``."""
    rows: list[dict[str, object]] = []
    for key, part in frame.groupby(keys, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, values)), **row_metrics(part)})
    return pd.DataFrame(rows)


def holm_adjust(values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values in the original order."""
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def paired_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    active: str = PUBLIC_METHODS[-1],
    references: tuple[str, ...] = PUBLIC_METHODS[:3],
    seed: int = 17,
    replicates: int = 5000,
) -> pd.DataFrame:
    """Compare interval methods by resampling evaluation units as clusters."""
    rows: list[dict[str, object]] = []
    keys = ["seed", "outer_fold", "sample_row"]
    for offset, (dataset, frame) in enumerate(
        predictions.groupby("dataset", sort=True)
    ):
        active_frame = (
            frame.loc[frame.method.eq(active)].sort_values(keys).reset_index(drop=True)
        )
        cluster_codes, clusters = pd.factorize(
            active_frame.unit_id.astype(str), sort=True
        )
        cluster_n = len(clusters)
        draws = np.random.default_rng(seed + offset).multinomial(
            cluster_n,
            np.full(cluster_n, 1 / cluster_n),
            size=replicates,
        )
        for reference in references:
            reference_frame = (
                frame.loc[frame.method.eq(reference)]
                .sort_values(keys)
                .reset_index(drop=True)
            )
            np.testing.assert_array_equal(active_frame[keys], reference_frame[keys])
            np.testing.assert_array_equal(
                active_frame.observed, reference_frame.observed
            )

            def values(part: pd.DataFrame) -> dict[str, np.ndarray]:
                observed = part.observed.to_numpy()
                lower = part.lower.to_numpy()
                upper = part.upper.to_numpy()
                return {
                    "coverage": ((observed >= lower) & (observed <= upper)).astype(
                        float
                    ),
                    "mean_width": upper - lower,
                    "mean_interval_score": interval_score(
                        observed, lower, upper, alpha=0.1
                    ),
                }

            active_values = values(active_frame)
            reference_values = values(reference_frame)
            for metric in (
                "coverage",
                "high_coverage",
                "mean_width",
                "mean_interval_score",
            ):
                take = (
                    active_frame.group_high_rest.eq("high").to_numpy()
                    if metric == "high_coverage"
                    else np.ones(len(active_frame), dtype=bool)
                )
                source = "coverage" if metric == "high_coverage" else metric
                difference = (active_values[source] - reference_values[source])[take]
                counts = np.bincount(cluster_codes[take], minlength=cluster_n)
                sums = np.bincount(
                    cluster_codes[take], weights=difference, minlength=cluster_n
                )
                denominator = draws @ counts
                if np.any(denominator == 0):
                    raise ValueError("empty cluster-bootstrap resample")
                bootstrap = (draws @ sums) / denominator
                point = float(difference.mean())
                p_value = (1 + int(np.sum(np.abs(bootstrap - point) >= abs(point)))) / (
                    len(bootstrap) + 1
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "active": active,
                        "reference": reference,
                        "metric": metric,
                        "point_difference": point,
                        "ci95_low": float(np.quantile(bootstrap, 0.025)),
                        "ci95_high": float(np.quantile(bootstrap, 0.975)),
                        "p_two_sided": p_value,
                        "bootstrap_replicates": replicates,
                        "unit_n": int(np.sum(counts > 0)),
                        "p_method": "null-centered paired cluster bootstrap",
                    }
                )

    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby(["dataset", "metric"])["p_two_sided"].transform(
        lambda values: holm_adjust(values.to_numpy())
    )
    result["holm_reject_0_05"] = result.p_holm < 0.05
    return result
