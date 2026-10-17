# Mondrian RCP-AbsRes for soil spectroscopy


## Current experiments

| Task | Samples | Evaluation |
|---|---:|---|
| LUCAS 2015 SOC | 21,782 | Four-method ablation, CQR and NCP |
| LUCAS 2015 TN | 21,782 | Four-method ablation |
| LUCAS 2009 SOC | 18,739 | Four-method ablation, CQR and NCP |
| Austrian SOC | 2,112 | Four-method ablation, CQR and NCP |
| LUCAS 2009 clay | 18,739 | Four-method ablation |

The four methods are Split CP, RCP-AbsRes, Mondrian Split CP and Mondrian RCP-AbsRes.
Each task has five seeded five-fold spatial/grouped partitions. Nominal coverage is
90%; the high-prediction threshold is the 80th percentile of predictions on the
residual-learning set. Datasets are fitted separately; this is not trained-model
transfer.

## Install

Use Python 3.10 or 3.11 in a new virtual environment:

    python -m venv .venv

Activate it using .venv\Scripts\Activate.ps1 on Windows or
source .venv/bin/activate on Linux/macOS, then:

    python -m pip install .
    python scripts/build_current_results.py --check
    python scripts/update_result_manifest.py --check

See REPRODUCIBILITY.md for the platforms and scope actually validated.
The public package omits the development test suite and automated CI workflow.

## Run the current protocol

Prepare inputs according to DATA_PREPARATION.md. The default retrains one
preselected cell (seed 20260831, fold 0) for each of seven computational paths:

    python -m d2c_v2.current_protocol --prepared-data data/prepared --output outputs/check

Retrain all 25 cells for one task:

    python -m d2c_v2.current_protocol --prepared-data data/prepared --task 2015_TN --full-matrix --output outputs/tn_all

Tasks: 2015_SOC, 2015_TN, 2009_SOC, Austria_SOC, Clay, 2009_public, Austria_public.
The last two fit CQR/NCP independently; their Split CP and Mondrian RCP-AbsRes
comparison rows come from the corresponding SOC ablation. Only prepared spectra,
labels and spatial folds enter training. No archived predictions/models or internal
plans are read. Output directories must not already exist.

## Diagnostic and statistical reproduction

Add `--threshold-scan` to compute the 50th–90th percentile analyses for the two
LUCAS SOC tasks without refitting models at each threshold. The current runner
also calculates compact residual/rectified-score diagnostics and group mean
widths within each of the five scientific tasks.

    python -m d2c_v2.current_protocol --prepared-data data/prepared --threshold-scan --output outputs/check
    python -m d2c_v2.study_statistics --run-dir outputs/check --bootstrap all --output outputs/statistics

Use `--full-matrix` in the training command for the complete manuscript matrix.
See `docs/STATISTICAL_REPRODUCTION.md` for denominators, quantiles, bootstrap
random seeds and the distinction between a seven-cell check and full reproduction.
No figure drawing, layout/export or document-building scripts are included.

## Results and provenance

results/CURRENT_RESULTS.json is the relative-path current result index.
results/current/ contains the five-task pooled summary and public-baseline median
summary. Source tables retain their original values and precision. See
results/README.md for the table dictionary and docs/PACKAGE_SCOPE.md for the package
scope.

## Method sources

- RCP: https://proceedings.mlr.press/v267/plassier25a.html
- Mondrian regression: https://proceedings.mlr.press/v128/bostrom20a.html
- Split conformal regression: https://doi.org/10.1080/01621459.2017.1307116

## License and repository

The code is licensed under the MIT License; see LICENSE.
Third-party data remain subject to their providers' terms.
Raw data, prepared training data and sample-level prediction archives
are not redistributed in this repository.

Repository: https://github.com/LovSherry/Mondrian-RCP-AbsRes

For software citation metadata, see CITATION.cff.
