# Reproducibility — code release 0.2.1

## Current evidence

The LUCAS 2015 analyses use all 21,782 eligible samples. The TN predictor is
ExtraTrees and uses the same spectral-preparation protocol as SOC. LUCAS 2009 and
Austria SOC, LUCAS 2009 clay, and the CQR/NCP comparisons are fitted separately
within each dataset.

results/CURRENT_RESULTS.json identifies the compact source tables.
results/current/pooled_summary.csv combines the five current ablations without
changing source values. public_median_summary.csv preserves per-cell medians used
for the public comparison; do not substitute pooled means. SOC/TN widths are in
g kg^-1, clay widths in percentage points.

For Mondrian RCP-AbsRes on LUCAS 2015, pooled SOC high-group coverage is
0.9012328891306318 and width is 50.11038249882686. TN values are
0.901939410455557 and 3.6697657562158676. These are experimental reference results,
not universal coverage guarantees.

## Result integrity checks

    python scripts/update_result_manifest.py --check
    python scripts/build_current_results.py --check

These commands verify the released file hashes and rebuild the current summary
views in memory for comparison. They do not retrain models. The development test
suite and automated CI workflow are not included in this public package.

## Retraining

    python -m d2c_v2.current_protocol --prepared-data data/prepared --output outputs/retrained

This trains seven preselected cells from scratch. Add --full-matrix for all 25
cells per selected task. Use --task, --seed and --fold to select a recorded cell.
Each cell writes predictions.parquet, metrics.csv and SUCCESS.json.
The five scientific tasks also write aggregate numerical diagnostics under
`statistics/`. Add `--threshold-scan` for the LUCAS SOC grouping-threshold scan.
`d2c_v2.study_statistics` pools these outputs and runs the paired bootstrap;
see `docs/STATISTICAL_REPRODUCTION.md` for commands and statistical definitions.
RUN_REPORT.json records input hashes and environment versions. A nonzero exit code
means an incomplete/failed run even if earlier cells finished. There is no automatic
retry; inspect the error and use a new destination. No old result is overwritten.

## Independent-environment validation

Validation installs a built wheel in a new virtual environment with freshly
installed dependencies, outside the source tree. Training uses relocated prepared
data. References are read only by the subsequent comparison, never by training.

Compare method/sample identities, observed values, point predictions, interval
endpoints and group labels. Report exact equality separately from the predefined
absolute numerical tolerance of 1e-10. Identities and group labels must match
exactly; timing and serialization metadata are not expected to match.

Recalculating pooled/per-cell metrics from all saved predictions is result
reconstruction, not full-matrix retraining or an independent implementation of
the learning algorithms. Same-host new-environment verification does not establish
cross-machine or cross-operating-system identity. The validation report lists the
actual scope achieved.

## Numerical boundaries

The shared finite quantile rejects empty, non-finite or non-vector scores, invalid
alpha, and ranks outside the calibration sample. It does not clip ranks or silently
discard invalid observations. The current protocol requires at least 50 samples
in every calibration group. These changes leave valid 90% calculations unchanged.
Historical Mondrian helpers retain their small-group global fallback; the current
runner rejects such cases.

## Statistical extension validation on 15 October 2026

All seven preselected paths were retrained using the pinned Python 3.10
environment. Their observed values, point predictions and interval endpoints
matched the previous validation exactly. This was seven-cell retraining, not a
fresh full-matrix retraining.

The numerical diagnostics were recalculated for all 125 frozen scientific cells,
without fitting new models. The 3,250 residual-diagnostic rows matched the earlier
independently checked calculations; rectified-score quantiles agreed within
1e-10. The threshold scan covered 41 quantiles in each of 50 LUCAS SOC cells.
The three SOC public-comparison bootstrap tables and the TN/clay single-contrast
confidence intervals were reconstructed from frozen predictions, agreeing with
the archived results within 1e-10. These checks establish numerical reproduction
on the validated environment, not new experimental or cross-platform evidence.

## Execution note

Use `d2c_v2.current_protocol` as the public entry point. Output directories must be
new so that a run cannot silently mix results from different code versions.
