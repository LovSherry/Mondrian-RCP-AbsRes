# Aggregated research results

This directory contains the aggregated numerical results used to verify the tables
and claims in the accompanying study. It does not contain raw spectra, exact sampling
coordinates, sample-level predictions or fitted models.

## Current views

- `CURRENT_RESULTS.json`: machine-readable index that maps each analysis to its
  source table.
- `current/pooled_summary.csv`: pooled coverage, group coverage, mean interval width
  and mean interval score for the four-method ablation across the five property and
  dataset analyses.
- `current/public_median_summary.csv`: per-cell median results for Split CP, standard
  CQR, NCP and Mondrian RCP-AbsRes.

## Source tables

- `lucas2015_full_cohort/*_metrics_by_cell.csv`: one row per method, seed and outer
  fold for the LUCAS 2015 SOC and TN analyses.
- `lucas2015_full_cohort/*_pooled_summary.csv`: pooled summaries for the same analyses.
- `lucas2015_full_cohort/*_ablation_bootstrap.csv`: paired spatial-cluster bootstrap
  contrasts for the four-method ablation.
- `lucas2015_full_cohort/SOC_median_summary.csv` and
  `SOC_public_bootstrap.csv`: CQR/NCP comparison summaries and paired contrasts.
- `headline_results.csv`: LUCAS 2009 and Austrian SOC summaries.
- `property_extension/headline.csv`: LUCAS 2009 clay summaries.
- `unified_public_baselines/median_summary.csv` and
  `paired_group_bootstrap.csv`: CQR/NCP comparison summaries and paired contrasts for
  LUCAS 2009 and Austria.

## Variables and units

Coverage columns are proportions on the interval `[0, 1]`. SOC and TN interval
widths are in g kg^-1; clay interval widths are in percentage points. Interval scores
use the 90% nominal-coverage setting. Empty cells in the current public-baseline view
mean that a group-specific width or interval score was not summarized; they are not
zero values.

## Integrity and provenance

`MANIFEST.sha256` records the SHA-256 digest of every released result file. Rebuild
the two current views with `python scripts/build_current_results.py` and verify all
digests with `python scripts/update_result_manifest.py --check`.

The source spectra and measured soil properties remain subject to their providers'
terms and are not relicensed here. Before depositing this result package, the authors
must select and record a licence for the newly generated aggregate tables.

## Additional diagnostic summaries

`diagnostics/` contains final aggregate numerical summaries for the six-figure
manuscript. No raw spectra, individual observations/predictions, sample identifiers
or per-cell intermediate tables are included:

- `residual_diagnostics_median_iqr.csv`: calibration/evaluation residual summaries
  for the two LUCAS SOC analyses, on fixed D_fit prediction groups/deciles.
- `tail_gaps_median_iqr.csv`: medians/IQRs of within-cell high-minus-remaining-group
  raw and rectified score-tail differences.
- `thresholds_pooled.csv`: complete-evaluation-set metrics and high-group coverage
  for each of the 41 separately calibrated thresholds in the two LUCAS SOC tasks.
- `group_widths_pooled.csv`: pooled group/overall mean widths for all five tasks
  under the common 80th-percentile reference.

Reproduction commands and denominator definitions are in
`docs/STATISTICAL_REPRODUCTION.md`.

