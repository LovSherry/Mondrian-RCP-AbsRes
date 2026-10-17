# Statistical reproduction

The entry points below reproduce numerical analyses without generating figures
or editing manuscripts. Obtain the third-party datasets under their providers'
terms and prepare them as described in `DATA_PREPARATION.md`.

## 1. Fit models and calculate cell statistics

Run the seven preselected cells, including the LUCAS SOC threshold scans:

    python -m d2c_v2.current_protocol --prepared-data data/prepared --threshold-scan --output outputs/check

For every analysis cell used in the manuscript, add `--full-matrix`:

    python -m d2c_v2.current_protocol --prepared-data data/prepared --full-matrix --threshold-scan --output outputs/full

There are five scientific tasks and two additional CQR/NCP computational paths.
Each complete task/path has five seeds and five folds. A seven-cell run is an
execution check, not a reconstruction of the full 25-cell summaries.

Within each cell, `statistics/` contains aggregate residual diagnostics, high-minus-
remaining-group tail differences and group mean widths. LUCAS SOC cells also
write results for all 41 grouping quantiles from 0.50 to 0.90. Point and residual-
quantile models remain fixed during this scan; both Mondrian methods are
recalibrated separately at each threshold using the calibration set.

## 2. Pool the results and calculate uncertainty

    python -m d2c_v2.study_statistics --run-dir outputs/full --bootstrap all --output outputs/statistics

Use `outputs/check` instead for a seven-cell execution check. The statistics
report records actual cell counts and does not label partial runs as complete
matrices. Existing output folders are rejected. Prediction and cell-statistic
hashes are checked against the corresponding successful training records.

| Manuscript quantity | Generated table |
| --- | --- |
| Figure 4 threshold curves | `thresholds_pooled.csv` |
| Figure 5 residual diagnostics | `diagnostics_median_iqr.csv` |
| Within-cell raw/rectified score-tail differences | `tail_gaps_median_iqr.csv` |
| Group mean widths and overall width | `group_widths_pooled.csv` |
| Pooled ablation metrics | `pooled_metrics.csv` |
| Per-cell medians for the established-method comparison | `cell_median_metrics.csv` |
| SOC paired comparisons with Split CP, CQR and NCP | `public_paired_bootstrap.csv` |
| Single contrast with Mondrian Split CP | `ablation_paired_bootstrap.csv` |

The release includes only compact final diagnostic/threshold/width summaries
under `results/diagnostics/`, not the generated per-cell tables or predictions.

## Definitions that affect reproduction

- Diagnostic groups and deciles are defined by the point-prediction distribution
  on D_fit. The same boundaries are applied to D_cal and D_eval. Ties are assigned
  to the upper bin; evaluation groups are not forced to contain exactly 20%.
- Signed residuals are observed minus predicted values. Centered residuals
  subtract the median signed residual separately within each group and partition.
- Empirical quantiles use linear interpolation. Conformal calibration uses
  `ceil((n + 1) * 0.9)` as the order-statistic rank, without linear interpolation.
- Distributional statistics and high-minus-remaining-group contrasts are first
  calculated within a cell, then summarized across the 25 cells. A difference
  between two medians is not substituted for the median paired difference.
- Overall coverage, width and interval score use the complete evaluation set.
  High-group coverage and width use the corresponding high-group denominator.
  Pooled group widths are weighted by group prediction-record counts, not by
  equally weighting folds or averaging the two group means.
- Bootstrap sampling uses complete physical spatial blocks or experimental
  groups. All five repeated predictions for a sample remain with its unit. Models
  are not refitted during resampling. Intervals therefore describe uncertainty
  conditional on the saved fitted predictions.
- SOC public comparisons use 5,000 paired replicates, null-centered two-sided
  approximate p values and Holm adjustment over three comparators within each
  dataset and metric. The single Mondrian Split CP contrast uses an unadjusted
  percentile interval, without importing that three-comparator correction.

## Bootstrap random streams

Defaults preserve the streams used for the reported numerical results:

| Analysis | Seed |
| --- | ---: |
| LUCAS 2015 SOC public comparisons | 20260916 |
| LUCAS 2009 SOC public comparisons | 20260917 |
| Austrian SOC public comparisons | 20260916 |
| SOC/TN single ablation contrast | 20260908 |
| Clay single ablation contrast | 20260907 |

`--bootstrap-seed` explicitly overrides these defaults; a changed seed can change
Monte Carlo confidence-interval endpoints. `--bootstrap-replicates` defaults to
5,000. These settings never alter fitted prediction intervals.

## Data and package boundaries

Local retraining writes per-sample predictions and, for some paths, calibration
arrays under the chosen output directory. They are needed for local calculation
but are not distributed. Do not upload these outputs, prepared inputs, models or
calibration arrays. Figure drawing/export, document production and packaging
scripts are deliberately excluded from the research code release.
