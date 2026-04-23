# Reproducibility

## Recommended runtime
Use a notebook environment with the dependencies in `requirements.txt`. The baseline training notebook was originally run in a Kaggle-style environment, while the final analysis notebooks can be inspected locally from the repository artifacts.

## Notebook order
1. `regression_model/ved-energy-regression.ipynb`
2. `notebooks/05_blogpost_main_flow.ipynb`
3. `notebooks/06_shap_vs_lime.ipynb`
4. `notebooks/07_trustworthiness_audit.ipynb`

## What each notebook does

### 1. `regression_model/ved-energy-regression.ipynb`
Trains or reloads the baseline XGBoost energy regressor and produces the scored trip table.

### 2. `notebooks/05_blogpost_main_flow.ipynb`
Builds the main tutorial figures:
- pipeline diagram
- predicted vs actual plot
- residual distribution
- predictive summary tables

### 3. `notebooks/06_shap_vs_lime.ipynb`
Generates the explanation section:
- manual SHAP global summary
- local SHAP plots
- local LIME plots
- one SHAP-vs-LIME comparison figure

### 4. `notebooks/07_trustworthiness_audit.ipynb`
Runs the trustworthiness section:
- leakage ablation
- validation-only threshold calibration
- LIME stability
- subgroup robustness

## Final output locations

### Main figures
`outputs_blogpost/figures/`
- `pipeline_diagram.png`
- `predicted_vs_actual_test.png`
- `residual_distribution_test.png`

### XAI outputs
`outputs_blogpost/xai/`
- `shap/shap_summary.png`
- `shap/trip_<ID>_shap.png`
- `lime/trip_<ID>_lime.png`
- `comparisons/shap_vs_lime_560_141.png`

### Audit outputs
`outputs_blogpost/audit/`
- `figures/leakage_ablation.png`
- `figures/lime_stability.png`
- `figures/subgroup_transmission_anomaly_rate.png`
- `figures/validation_threshold_calibration.png`
- `tables/*.csv`

### Summary tables
`outputs_blogpost/tables/`
- `predictive_summary.csv`
- `split_summary.csv`
- `top_test_anomalies.csv`
- `xai_case_summary.csv`
- `lime_case_summary.csv`
- `shap_case_summary.csv`

## Important interpretation note
This project is a **decision-support tutorial**, not a direct diagnostic tool. A flagged trip means the trip is unexpected under the model, not that the vehicle definitely has a fault.
