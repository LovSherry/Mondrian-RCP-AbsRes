from __future__ import annotations

from typing import Any

import numpy as np

from d2c_v2.modeling import hist_quantile


def group_labels(values: np.ndarray, threshold: float) -> np.ndarray:
    """Assign high/rest labels from a threshold fixed on D_fit predictions."""

    return np.where(np.asarray(values, dtype=float) >= float(threshold), "high", "rest")


def fit_tau_model(
    x_train: np.ndarray,
    score_train: np.ndarray,
    arrays: list[np.ndarray],
    *,
    nominal: float,
    config: dict[str, Any],
    seed: int,
) -> list[np.ndarray]:
    """Fit the RCP conditional score-quantile model on D_fit only."""

    model = hist_quantile(config, float(nominal), seed)
    model.fit(
        np.asarray(x_train, dtype=np.float32), np.asarray(score_train, dtype=float)
    )
    output = [
        np.asarray(model.predict(np.asarray(value, dtype=np.float32)), dtype=float)
        for value in arrays
    ]
    if not all(np.isfinite(value).all() for value in output):
        raise RuntimeError("tau model produced non-finite output")
    return output
