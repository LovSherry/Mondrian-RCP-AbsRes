"""Build current, method-ordered views without altering historical source tables."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "2015_SOC": "results/lucas2015_full_cohort/SOC_pooled_summary.csv",
    "2015_TN": "results/lucas2015_full_cohort/TN_pooled_summary.csv",
    "2009_Austria_SOC": "results/headline_results.csv",
    "Clay": "results/property_extension/headline.csv",
    "2015_public_medians": "results/lucas2015_full_cohort/SOC_median_summary.csv",
    "2009_Austria_public_medians": (
        "results/unified_public_baselines/median_summary.csv"
    ),
}
METHODS = ["Split CP", "RCP-AbsRes", "Mondrian Split CP", "Mondrian RCP-AbsRes"]
DIAGNOSTICS = {
    "residuals": "results/diagnostics/residual_diagnostics_median_iqr.csv",
    "score_tail_gaps": "results/diagnostics/tail_gaps_median_iqr.csv",
    "group_widths": "results/diagnostics/group_widths_pooled.csv",
    "threshold_scan": "results/diagnostics/thresholds_pooled.csv",
}


def outputs():
    rows = []
    for target in ("SOC", "TN"):
        table = pd.read_csv(ROOT / SOURCES[f"2015_{target}"])
        rows.append(
            table.loc[table.method.isin(METHODS)].assign(
                dataset="lucas2015", target=target
            )
        )
    table = pd.read_csv(ROOT / SOURCES["2009_Austria_SOC"])
    table = table.loc[table.dataset.isin(["lucas2009", "austrian"])].copy()
    table = table.rename(
        columns={
            "overall_coverage": "coverage",
            "interval_score": "mean_interval_score",
        }
    )
    rows.append(table.assign(target="SOC"))
    clay = pd.read_csv(ROOT / SOURCES["Clay"]).query("target=='Clay'").copy()
    clay = clay.rename(columns={"n": "n_predictions", "unique_samples": "n_unique"})
    rows.append(clay)
    pooled = pd.concat(rows, ignore_index=True)
    cols = [
        "dataset",
        "target",
        "method",
        "coverage",
        "high_coverage",
        "rest_coverage",
        "mean_width",
        "mean_interval_score",
    ]
    pooled["order"] = pooled.method.map({m: i for i, m in enumerate(METHODS)})
    pooled = pooled.sort_values(["order", "dataset", "target"])[cols]
    median = pd.read_csv(ROOT / SOURCES["2009_Austria_public_medians"])
    median = median.loc[
        median.dataset.isin(["lucas2009", "austrian"]) & median.nominal_coverage.eq(0.9)
    ].copy()
    current = pd.read_csv(ROOT / SOURCES["2015_public_medians"])
    extra = []
    for _, row in current.loc[
        current.method.isin(["Split CP", "Standard CQR", "NCP", "Mondrian RCP-AbsRes"])
    ].iterrows():
        for subset, column in [
            ("overall", "coverage"),
            ("high", "high_coverage"),
            ("rest", "rest_coverage"),
        ]:
            # Width and score remain overall medians; subgroup values are not implied.
            extra.append(
                dict(
                    dataset="lucas2015",
                    method=row.method,
                    nominal_coverage=0.9,
                    subset=subset,
                    coverage=row[column],
                    mean_width=row.mean_width if subset == "overall" else None,
                    mean_interval_score=row.mean_interval_score
                    if subset == "overall"
                    else None,
                )
            )
    median = pd.concat([median, pd.DataFrame(extra)], ignore_index=True)
    median = median[
        [
            "dataset",
            "method",
            "nominal_coverage",
            "subset",
            "coverage",
            "mean_width",
            "mean_interval_score",
        ]
    ]
    median = median.sort_values(["method", "dataset", "subset"])
    index = dict(
        study=(
            "Mondrian rectified conformal prediction for conditional reliability "
            "in soil spectroscopic regression"
        ),
        code_version="0.2.1",
        sources=SOURCES,
        derived_views=[
            "results/current/pooled_summary.csv",
            "results/current/public_median_summary.csv",
        ],
        diagnostic_summaries=DIAGNOSTICS,
        note=(
            "Views preserve source precision; overall widths are not high-group widths."
        ),
    )
    return {
        "results/CURRENT_RESULTS.json": json.dumps(index, indent=2) + "\n",
        "results/current/pooled_summary.csv": pooled.to_csv(
            index=False, lineterminator="\n"
        ),
        "results/current/public_median_summary.csv": median.to_csv(
            index=False, lineterminator="\n"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for rel, text in outputs().items():
        path = ROOT / rel
        if args.check:
            if not path.exists() or path.read_text(encoding="utf8") != text:
                raise ValueError(f"current view missing or stale: {rel}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf8", newline="\n")
    print(
        "Current result views verified"
        if args.check
        else "Current result views written"
    )


if __name__ == "__main__":
    main()
