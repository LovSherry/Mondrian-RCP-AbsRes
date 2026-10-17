from __future__ import annotations

from typing import Any

import numpy as np

from d2c_v2.modeling import conformal_quantile
from d2c_v2.rcp_absres import absolute_residual_score


def _validated_groups(
    group_cal: np.ndarray,
    group_eval: np.ndarray,
    calibration_n: int,
    evaluation_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.asarray(group_cal, dtype=str)
    evaluation = np.asarray(group_eval, dtype=str)
    if len(calibration) != int(calibration_n) or len(evaluation) != int(evaluation_n):
        raise ValueError("Mondrian group arrays are unaligned")
    if not set(np.unique(evaluation)).issubset(set(np.unique(calibration))):
        raise ValueError(
            "evaluation contains a Mondrian category absent from calibration"
        )
    return calibration, evaluation


def group_and_global_quantiles(
    scores: np.ndarray,
    group_cal: np.ndarray,
    *,
    alpha: float,
    minimum_group_n: int,
) -> tuple[float, dict[str, float], dict[str, int]]:
    values = np.asarray(scores, dtype=float)
    groups = np.asarray(group_cal, dtype=str)
    if len(values) != len(groups):
        raise ValueError("scores and calibration groups are unaligned")
    if not np.isfinite(values).all():
        raise ValueError("calibration scores contain non-finite values")
    global_quantile = float(conformal_quantile(values, float(alpha)))
    by_group: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label in sorted(np.unique(groups)):
        take = groups == label
        count = int(take.sum())
        counts[label] = count
        by_group[label] = (
            float(conformal_quantile(values[take], float(alpha)))
            if count >= int(minimum_group_n)
            else global_quantile
        )
    return global_quantile, by_group, counts


def apply_mondrian_split_cp(
    *,
    nominal: float,
    observed_cal: np.ndarray,
    point_cal: np.ndarray,
    point_eval: np.ndarray,
    group_cal: np.ndarray,
    group_eval: np.ndarray,
    minimum_group_n: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    score = absolute_residual_score(observed_cal, point_cal)
    point = np.asarray(point_eval, dtype=float)
    calibration_groups, evaluation_groups = _validated_groups(
        group_cal, group_eval, len(score), len(point)
    )
    global_quantile, by_group, counts = group_and_global_quantiles(
        score,
        calibration_groups,
        alpha=1.0 - float(nominal),
        minimum_group_n=int(minimum_group_n),
    )
    radius = np.asarray([by_group[label] for label in evaluation_groups], dtype=float)
    lower, upper = point - radius, point + radius
    if not np.isfinite(np.column_stack([lower, upper])).all() or np.any(lower > upper):
        raise RuntimeError("Mondrian Split CP produced an invalid interval")
    return (
        lower,
        upper,
        {
            "q_global": global_quantile,
            "q_by_group": by_group,
            "calibration_group_n": counts,
            "minimum_group_n": int(minimum_group_n),
        },
    )


def apply_mondrian_rcp_absres(
    *,
    nominal: float,
    observed_cal: np.ndarray,
    point_cal: np.ndarray,
    point_eval: np.ndarray,
    tau_cal: np.ndarray,
    tau_eval: np.ndarray,
    q0: float,
    group_cal: np.ndarray,
    group_eval: np.ndarray,
    minimum_group_n: int,
    selected_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    score = absolute_residual_score(observed_cal, point_cal)
    point = np.asarray(point_eval, dtype=float)
    tau_calibration = np.asarray(tau_cal, dtype=float)
    tau_evaluation = np.asarray(tau_eval, dtype=float)
    if len(tau_calibration) != len(score) or len(tau_evaluation) != len(point):
        raise ValueError("RCP tau arrays are unaligned")
    if not np.isfinite(np.concatenate([tau_calibration, tau_evaluation, point])).all():
        raise ValueError("RCP inputs contain non-finite values")
    calibration_groups, evaluation_groups = _validated_groups(
        group_cal, group_eval, len(score), len(point)
    )
    lam = float(selected_lambda)
    tau_lambda_cal = float(q0) + lam * (tau_calibration - float(q0))
    tau_lambda_eval = float(q0) + lam * (tau_evaluation - float(q0))
    rectified = score - tau_lambda_cal
    global_quantile, by_group, counts = group_and_global_quantiles(
        rectified,
        calibration_groups,
        alpha=1.0 - float(nominal),
        minimum_group_n=int(minimum_group_n),
    )
    correction = tau_lambda_eval + np.asarray(
        [by_group[label] for label in evaluation_groups], dtype=float
    )
    radius = np.maximum(correction, 0.0)
    lower, upper = point - radius, point + radius
    if not np.isfinite(np.column_stack([lower, upper])).all() or np.any(lower > upper):
        raise RuntimeError("Mondrian RCP-AbsRes produced an invalid interval")
    return (
        lower,
        upper,
        {
            "q0": float(q0),
            "lambda": lam,
            "q_global": global_quantile,
            "q_by_group": by_group,
            "calibration_group_n": counts,
            "minimum_group_n": int(minimum_group_n),
            "nonnegative_clip_rate": float(np.mean(correction < 0.0)),
        },
    )
