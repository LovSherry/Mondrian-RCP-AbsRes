from __future__ import annotations

from typing import Any

import numpy as np

from d2c_v2.modeling import conformal_quantile


def absolute_residual_score(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if observed_array.shape != predicted_array.shape:
        raise ValueError("observed and predicted arrays must be aligned")
    score = np.abs(observed_array - predicted_array)
    if not np.isfinite(score).all():
        raise ValueError("absolute-residual score contains non-finite values")
    return score


def apply_rcp_absres(
    *,
    nominal: float,
    observed_cal: np.ndarray,
    point_cal: np.ndarray,
    point_eval: np.ndarray,
    tau_cal: np.ndarray,
    tau_eval: np.ndarray,
    q0: float,
    selected_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    alpha = 1.0 - float(nominal)
    score_cal = absolute_residual_score(observed_cal, point_cal)
    point_eval_array = np.asarray(point_eval, dtype=float)
    tau_cal_array = np.asarray(tau_cal, dtype=float)
    tau_eval_array = np.asarray(tau_eval, dtype=float)
    if len(tau_cal_array) != len(score_cal):
        raise ValueError("calibration tau is unaligned")
    if len(tau_eval_array) != len(point_eval_array):
        raise ValueError("evaluation tau is unaligned")
    if not np.isfinite(
        np.concatenate([tau_cal_array, tau_eval_array, point_eval_array])
    ).all():
        raise ValueError("RCP absolute-residual inputs contain non-finite values")

    lam = float(selected_lambda)
    tau_lambda_cal = float(q0) + lam * (tau_cal_array - float(q0))
    tau_lambda_eval = float(q0) + lam * (tau_eval_array - float(q0))
    rectified_score = score_cal - tau_lambda_cal
    rectified_quantile = conformal_quantile(rectified_score, alpha)
    unbounded_radius = tau_lambda_eval + rectified_quantile
    radius = np.maximum(unbounded_radius, 0.0)
    lower = point_eval_array - radius
    upper = point_eval_array + radius
    if not np.isfinite(np.column_stack([lower, upper])).all() or np.any(lower > upper):
        raise RuntimeError("RCP absolute-residual interval is invalid")
    metadata = {
        "q0": float(q0),
        "lambda": lam,
        "rectified_calibration_quantile": float(rectified_quantile),
        "nonnegative_clip_rate": float(np.mean(unbounded_radius < 0.0)),
    }
    return lower, upper, metadata


def split_cp_interval(
    *,
    nominal: float,
    observed_cal: np.ndarray,
    point_cal: np.ndarray,
    point_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    alpha = 1.0 - float(nominal)
    q = conformal_quantile(absolute_residual_score(observed_cal, point_cal), alpha)
    point = np.asarray(point_eval, dtype=float)
    return point - q, point + q, float(q)
