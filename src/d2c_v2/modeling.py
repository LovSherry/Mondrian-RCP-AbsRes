from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def point_model(
    kind: str,
    parameter: int | None,
    seed: int,
    *,
    extra_trees_n_jobs: int = 1,
) -> Any:
    """Build the frozen point-prediction family used in the paper experiments."""

    if kind != "ExtraTrees":
        raise ValueError(
            f"unsupported point model in the public paper workflow: {kind}"
        )
    return ExtraTreesRegressor(
        n_estimators=256,
        min_samples_leaf=3,
        max_features=0.70,
        n_jobs=int(extra_trees_n_jobs),
        random_state=int(seed),
    )


def make_pca_features(
    x_train: np.ndarray,
    arrays: list[np.ndarray],
    *,
    components: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray], Any]:
    from sklearn.decomposition import PCA

    count = max(2, min(int(components), x_train.shape[1], len(x_train) - 1))
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=count, svd_solver="randomized", random_state=int(seed)
                ),
            ),
        ]
    )
    train = np.asarray(pipe.fit_transform(x_train), dtype=np.float32)
    transformed = [
        np.asarray(pipe.transform(value), dtype=np.float32) for value in arrays
    ]
    return train, transformed, pipe


def hist_quantile(config: dict[str, Any], quantile: float, seed: int):
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        max_iter=int(config["tau_max_iter"]),
        max_leaf_nodes=int(config["tau_max_leaf_nodes"]),
        min_samples_leaf=int(config["tau_min_samples_leaf"]),
        learning_rate=float(config["tau_learning_rate"]),
        l2_regularization=float(config["tau_l2_regularization"]),
        random_state=int(seed),
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=10,
    )


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return a finite conformal order statistic; reject unsupported inputs."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a nonempty finite vector")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    rank = int(math.ceil((len(values) + 1) * (1.0 - float(alpha))))
    if not 1 <= rank <= len(values):
        raise ValueError(
            "insufficient calibration samples for a finite conformal quantile"
        )
    return float(np.partition(values, rank - 1)[rank - 1])


def cqr_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """CQR conformity score, used only by the supplementary public baseline."""

    return np.maximum(
        np.asarray(lower) - np.asarray(y), np.asarray(y) - np.asarray(upper)
    )


def interval_score(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return (
        hi - lo + 2.0 / alpha * (lo - y) * (y < lo) + 2.0 / alpha * (y - hi) * (y > hi)
    )
