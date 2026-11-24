# Results

Copy the contents of `MyDrive/nlbse27_results/` (written by the notebook) into this folder before tagging the release. Expected files:

- `results_main.csv`: all approaches on the test and clean-test splits (Table I)
- `statistical_tests.csv`: McNemar tests and bootstrap intervals
- `ablation_runs.csv`, `ablation_summary.csv`, `ablation_*_seed*.json`: ablation (Table II, Fig. 1)
- `error_rates_by_subset.csv`, `error_analysis_sample.csv`: error analysis
- `cost.csv`, `timing.json`: runtime and compute
- `eda_summary.csv`, `eda_satd_keywords.csv`, `eda.png`: exploratory analysis
- `confusion_matrices.png`, `thresholds.json`

The large `.npy` prediction caches can be left out or added if under 100 MB.
