"""Summarize completed runs and reproduce paired cluster-bootstrap statistics.

Only local run outputs are read. Published outputs are aggregate tables; raw,
prepared and per-sample prediction data are never copied to the result folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from d2c_v2.protocol_utils import PUBLIC_METHODS, paired_cluster_bootstrap, summaries


def protocol_bootstrap_seed(dataset: str, target: str, family: str) -> int:
    """Retain the random streams used for the reported comparisons."""
    if family == "public":
        return {"lucas2015": 20260916, "lucas2009": 20260917, "austrian": 20260916}[
            dataset
        ]
    return 20260907 if target == "Clay" else 20260908


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median_iqr(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Summarize cell statistics; do not calculate contrasts after summarizing."""
    numeric = [
        c
        for c in frame.select_dtypes(include=np.number)
        if c not in {*keys, "seed", "fold"}
    ]
    records = []
    for key, part in frame.groupby(keys, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        record = {**dict(zip(keys, values)), "cells": len(part)}
        for column in numeric:
            for p, suffix in ((0.25, "q25"), (0.5, "median"), (0.75, "q75")):
                record[column + "_" + suffix] = float(
                    part[column].quantile(p, interpolation="linear")
                )
        records.append(record)
    return pd.DataFrame(records)


def pooled_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool cell metrics with the correct overall/high/rest denominators."""
    rows = []
    for key, part in frame.groupby(["task", "high_quantile", "method"], sort=True):
        row = dict(zip(["task", "high_quantile", "method"], key))
        row.update(
            cells=len(part),
            n_predictions=int(part.n.sum()),
            n_high=int(part.n_high.sum()),
            n_rest=int(part.n_rest.sum()),
        )
        for column in (
            "coverage",
            "mean_width",
            "mean_interval_score",
            "high_coverage",
            "high_mean_width",
            "rest_coverage",
            "rest_mean_width",
        ):
            denominator = (
                "n_high"
                if column.startswith("high_")
                else "n_rest"
                if column.startswith("rest_")
                else "n"
            )
            row[column] = float(np.average(part[column], weights=part[denominator]))
        rows.append(row)
    return pd.DataFrame(rows)


def validate_pairing(frame: pd.DataFrame, methods: tuple[str, ...]) -> None:
    """Reject duplicate, mismatched or differently grouped evaluation records."""
    keys = ["seed", "outer_fold", "sample_row"]
    anchor = None
    for method in methods:
        part = (
            frame.loc[frame.method.eq(method)].sort_values(keys).reset_index(drop=True)
        )
        if part.empty or part.duplicated(keys).any():
            raise ValueError("missing method or duplicate evaluation records")
        if anchor is None:
            anchor = part
        else:
            for columns in (
                keys,
                ["unit_id", "group_high_rest"],
                ["observed", "predicted"],
            ):
                if not part[columns].equals(anchor[columns]):
                    raise ValueError(
                        "paired methods differ in records, units, groups or point predictions"
                    )


def summarize_runs(
    run_dir: Path,
    output: Path,
    *,
    bootstrap: str = "none",
    seed: int | None = None,
    replicates: int = 5000,
) -> dict:
    """Reconstruct aggregate results without re-fitting any model.

    The bootstrap resamples whole physical units, retaining their repeated
    predictions. Its uncertainty is conditional on the saved fitted predictions.
    """
    if output.exists():
        raise FileExistsError("choose a new statistics output directory")
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates required")
    report = json.loads((run_dir / "RUN_REPORT.json").read_text(encoding="utf8"))
    if report.get("status") != "COMPLETED" or not report.get("records"):
        raise ValueError("a completed nonempty run report is required")
    output.mkdir(parents=True)
    diagnostic_parts, gap_parts, scan_parts, width_parts, prediction_parts = (
        [],
        [],
        [],
        [],
        [],
    )
    cell_counts = {}
    seen = set()
    for record in report["records"]:
        task, seed_value, fold = (
            record["task"],
            int(record["seed"]),
            int(record["fold"]),
        )
        if not isinstance(task, str) or not task.replace("_", "").isalnum():
            raise ValueError("invalid task identifier")
        identity = task, seed_value, fold
        if seed_value not in range(20260831, 20260836) or fold not in range(5):
            raise ValueError("cell is outside the recorded five-seed/five-fold design")
        if identity in seen:
            raise ValueError("duplicate task/cell in run report")
        seen.add(identity)
        cell_counts[task] = cell_counts.get(task, 0) + 1
        folder = run_dir / f"{task}_{seed_value}_{fold}"
        if not (folder / "SUCCESS.json").is_file():
            raise ValueError("incomplete cell in completed run")
        success = json.loads((folder / "SUCCESS.json").read_text(encoding="utf8"))
        if success["prediction_sha256"] != _file_hash(folder / "predictions.parquet"):
            raise ValueError("prediction file changed after successful training")
        frame = pd.read_parquet(folder / "predictions.parquet")
        prediction_parts.append(frame)
        for name, destination in (
            ("residual_diagnostics.csv", diagnostic_parts),
            ("tail_gaps.csv", gap_parts),
            ("threshold_metrics.csv", scan_parts),
            ("group_widths.csv", width_parts),
        ):
            path = folder / "statistics" / name
            if path.is_file():
                expected = success.get("statistics_files", {}).get(name)
                if expected is not None and _file_hash(path) != expected:
                    raise ValueError(
                        "cell statistics changed after successful training"
                    )
                part = pd.read_csv(path)
                if (
                    not part.task.eq(task).all()
                    or not part.seed.eq(seed_value).all()
                    or not part.fold.eq(fold).all()
                ):
                    raise ValueError(
                        "statistics cell identifiers do not match run report"
                    )
                destination.append(part)
    tables = {}
    if diagnostic_parts:
        diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
        tables["diagnostics_all_cells.csv"] = diagnostics
        tables["diagnostics_median_iqr.csv"] = median_iqr(
            diagnostics, ["task", "partition", "group"]
        )
    if gap_parts:
        gaps = pd.concat(gap_parts, ignore_index=True)
        tables["tail_gaps_all_cells.csv"] = gaps
        tables["tail_gaps_median_iqr.csv"] = median_iqr(gaps, ["task", "partition"])
    for name, parts in (("thresholds", scan_parts), ("group_widths", width_parts)):
        if parts:
            cells = pd.concat(parts, ignore_index=True)
            tables[name + "_all_cells.csv"] = cells
            tables[name + "_pooled.csv"] = pooled_thresholds(cells)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    keys = ["dataset", "target", "seed", "outer_fold", "method", "sample_row"]
    if predictions.duplicated(keys).any():
        raise ValueError("duplicated method/sample records across computational paths")
    tables["pooled_metrics.csv"] = summaries(
        predictions, ["dataset", "target", "method"]
    )
    cell_metrics = summaries(
        predictions, ["dataset", "target", "seed", "outer_fold", "method"]
    )
    tables["cell_median_metrics.csv"] = (
        cell_metrics.groupby(["dataset", "target", "method"], as_index=False)
        .median(numeric_only=True)
        .drop(columns=["seed", "outer_fold"])
    )
    public_results, ablation_results, bootstrap_seeds = [], [], {}
    for (dataset, target), part in predictions.groupby(
        ["dataset", "target"], sort=True
    ):
        available = set(part.method)
        if bootstrap in {"public", "all"} and set(PUBLIC_METHODS) <= available:
            validate_pairing(part, PUBLIC_METHODS)
            chosen_seed = (
                protocol_bootstrap_seed(dataset, target, "public")
                if seed is None
                else seed
            )
            bootstrap_seeds[f"{dataset}/{target}/public"] = chosen_seed
            result = paired_cluster_bootstrap(
                part, seed=chosen_seed, replicates=replicates
            )
            public_results.append(result.assign(target=target))
        if (
            bootstrap in {"ablation", "all"}
            and {"Mondrian Split CP", "Mondrian RCP-AbsRes"} <= available
        ):
            validate_pairing(part, ("Mondrian Split CP", "Mondrian RCP-AbsRes"))
            chosen_seed = (
                protocol_bootstrap_seed(dataset, target, "ablation")
                if seed is None
                else seed
            )
            bootstrap_seeds[f"{dataset}/{target}/ablation"] = chosen_seed
            result = paired_cluster_bootstrap(
                part,
                references=("Mondrian Split CP",),
                seed=chosen_seed,
                replicates=replicates,
            )
            # Single percentile-interval contrast, not a three-test Holm family.
            result = result.drop(
                columns=["p_two_sided", "p_method", "p_holm", "holm_reject_0_05"]
            )
            ablation_results.append(
                result.assign(target=target, interval_kind="unadjusted_95_percentile")
            )
    if public_results:
        tables["public_paired_bootstrap.csv"] = pd.concat(
            public_results, ignore_index=True
        )
    if ablation_results:
        tables["ablation_paired_bootstrap.csv"] = pd.concat(
            ablation_results, ignore_index=True
        )
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    result = dict(
        status="COMPLETED",
        cells_per_task=cell_counts,
        full_matrix_per_task={task: count == 25 for task, count in cell_counts.items()},
        model_refits=0,
        bootstrap=bootstrap,
        bootstrap_seeds=bootstrap_seeds,
        bootstrap_replicates=replicates if bootstrap != "none" else 0,
        empirical_quantiles="linear interpolation within cell, then median across cells",
        pooled_metrics="all repeated out-of-fold predictions; sample repeats are not independent units",
        bootstrap_unit="complete spatial block or experimental group, retaining all repeated predictions",
        output_tables=sorted(tables),
    )
    (output / "STATISTICS_REPORT.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap", choices=("none", "public", "ablation", "all"), default="none"
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        help="override the documented dataset-specific seeds",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    result = summarize_runs(
        args.run_dir,
        args.output,
        bootstrap=args.bootstrap,
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
