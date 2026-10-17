# Prepared data for the current protocol

The training command consumes processed spectra rather than raw provider files.
Obtain third-party data under the providers' access and licence terms. The prepared
arrays are not redistributed in this repository.

## Folder contract

Under the prepared-data root, create lucas2015, lucas2009, austrian and clay.
Each folder contains:

| File | Content |
|---|---|
| features.npy | Finite row-aligned float32 matrix (n, p) |
| samples.parquet | sample_row, sample_id, unit_id, conditional_label and target columns |
| roles.parquet | seed, sample_row, unit_id, outer_fold for five saved partitions |
| dataset.json | n, p, targets mapping, features_sha256, seeds, folds |

sample_row is 0 through n-1 in order. Sample IDs must be unique. Each spatial block
or experiment stays in one fold. Fold labels are 0–4; seeds are 20260831–20260835.
The runner validates alignment and group separation.

dataset.json maps target names to sample-table columns, for example
"targets": {"SOC": "soc_g_kg", "TN": "tn_g_kg"}. features_sha256 must be the actual
SHA-256 digest of features.npy.

| Folder | n | p | Target units |
|---|---:|---:|---|
| lucas2015 | 21,782 | 196 | SOC/TN: g kg^-1 |
| lucas2009 | 18,739 | 196 | SOC: g kg^-1 |
| austrian | 2,112 | 183 | SOC: g kg^-1 |
| clay | 18,739 | 196 | Clay: mass percentage |

LUCAS unit_id denotes a 100-km ETRS-LAEA block; Austrian unit_id denotes the
experiment identifier. conditional_label is not a predictor.

## Preparation provenance

Preserve source/version identifiers, provider licences, checksums, exclusion counts,
unit conversions and row-alignment checks. Spectral processing uses the first
Savitzky–Golay derivative (about 20-nm window, polynomial order 2), a 10-nm modelling
grid, and per-spectrum SNV. Bands are 500–2450 nm for LUCAS and 680–2500 nm for
Austria. Recovery inputs already contain these transformations; do not repeat them.

The saved-fold contract is required for this manuscript. Use
d2c_v2.current_protocol rather than substituting a generic random train/test split.

## Training isolation

Two folds fit the point predictor and PCA. One learns residual corrections, one
calibrates intervals, and one evaluates them. CQR fits quantile models on the
point-training and residual-learning folds together. NCP learns residual scale
on the residual-learning fold. Final calibration/evaluation remain shared.
Evaluation labels do not determine thresholds, settings or model selection.
