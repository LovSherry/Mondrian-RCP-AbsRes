# Package scope

Release 0.2.1 is a focused distribution of the implementation and aggregated results
used in the study. Formatting and packaging changes do not alter numerical training
functions.

## Included

- The portable `d2c_v2.current_protocol` entry point and its transitive dependencies.
- Compact result tables and their SHA-256 manifest.
- Numerical residual diagnostics, threshold recalibration and paired-bootstrap
  entry points, with compact final statistical summaries.

Shared numerical modules support the documented current entry point. Use that
entry point rather than invoking helper modules as standalone experiments.

## Excluded

Prepared/raw data, fitted models, prediction archives, manuscript files, local
absolute-path configurations, caches and environment directories are not included.
Figure drawing/beautification, image export, figure-package assembly and Word/PDF
production scripts are also excluded. Numerical statistics used by the manuscript
are research code and remain included.
The development test suite and automated CI workflow are also excluded.

The package does not itself supply the prepared data required for training.
See DATA_PREPARATION.md for the input contract and provider rights.

## Reproduction claims

Rebuilding compact summary tables does not retrain models. The seven-cell
retraining command fits models from prepared inputs. The full-matrix
option fits all 25 cells for selected tasks. Reproduction does not establish
unconditional coverage guarantees or cross-platform bitwise equality.
