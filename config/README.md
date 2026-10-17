# Experiment settings

The portable entry point defines the retained experiment settings in
`src/d2c_v2/current_protocol.py`: `TASKS`, `SEEDS`, `HGB` and `ablation_config`.
It accepts input/output locations through command-line arguments. No local
machine path is required in a configuration file.

The point predictor uses 256 ExtraTrees, a minimum of three samples per leaf and
70% of spectral variables per split. Auxiliary quantile/scale regressors use
100 iterations, 15 leaves, minimum leaf size 40, learning rate 0.05 and L2 penalty 2.
There are five seeds and five folds. Nominal coverage is 90%; the high-prediction
threshold is fitted at the 80th percentile on the residual-learning set.

Changing these settings constitutes a new experiment and should use a new output
directory; it is not exact reproduction of the manuscript configuration.
