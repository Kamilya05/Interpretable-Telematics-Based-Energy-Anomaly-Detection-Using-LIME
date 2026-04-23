# VED Energy Baseline: `src/` Usage

This folder contains reusable code for the VED coursework project:

- detect anomalous trips using residuals (`actual - predicted`)
- generate LIME explanations for the model predictions behind selected trips
- run trustworthiness checks for calibration, leakage, stability, and subgroups
- save basic plots for reporting

## Assumptions

The baseline notebook has already produced:

- `regression_model/outputs/residuals.parquet` with `trip_id`, actual energy, predictions, and residuals
- `regression_model/models/xgb_energy_regressor.joblib` or `regression_model/models/xgb_energy_artifact.joblib`

For LIME explanations, use a feature-rich scored trip table. The compact residuals file is enough for anomaly ranking, but not enough to rebuild the model inputs needed to explain a prediction.

## Folder Structure

```text
src/
  anomaly/        residual thresholding and anomaly table generation
  audit/          calibration, leakage, stability, subgroup checks
  cli/            runnable scripts
  modeling/       feature design helpers
  utils/          I/O helpers, artifact loading, schema definitions
  viz/            helper plots
  xai/            LIME perturbation, kernel, surrogate, fidelity logic
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## 1. Detect Anomalies

Creates an anomalies table based on a residual threshold.

```bash
python -m src.cli.detect_anomalies \
  --residuals_path regression_model/outputs/residuals.parquet \
  --method quantile --quantile 0.98 \
  --out_anomalies_path outputs/anomalies.parquet \
  --out_meta_path outputs/anomaly_config.json
```

Outputs:

- `outputs/anomalies.parquet` - anomalous trips, ranked by residual
- `outputs/anomaly_config.json` - threshold metadata

Robust MAD-z thresholding is also available:

```bash
python -m src.cli.detect_anomalies \
  --residuals_path regression_model/outputs/residuals.parquet \
  --method mad_z --mad_z 3.5 \
  --out_anomalies_path outputs/anomalies.parquet \
  --out_meta_path outputs/anomaly_config.json
```

## 2. Explain A Trip With LIME

LIME explains the model's predicted `energy_per_km` for the selected trip. The residual table then shows whether the observed trip was unusually high relative to that prediction.

```bash
python -m src.cli.explain_trip \
  --residuals_path path/to/trip_table_scored_with_features.parquet \
  --model_path regression_model/models/xgb_energy_regressor.joblib \
  --trip_id <TRIP_ID> \
  --n_samples 5000 \
  --top_k 10
```

Outputs:

- `outputs/explanations/trip_<TRIP_ID>_lime.json`
- `outputs/explanations/trip_<TRIP_ID>_lime_bar.png`

Key JSON fields:

- `top_features`: top local contributions
- `local_r2`, `local_rmse`: surrogate fidelity near the instance
- `kernel_width`: proximity kernel width used by LIME

## 3. Basic Plots

```python
import pandas as pd
from src.viz.plots import plot_residual_hist, plot_actual_vs_pred

df = pd.read_parquet("regression_model/outputs/residuals.parquet")
plot_residual_hist(df, residual_col="residual", out_path="outputs/residual_hist.png")
plot_actual_vs_pred(
    df,
    y_col="energy_per_km",
    yhat_col="predicted_energy_per_km",
    out_path="outputs/actual_vs_pred.png",
)
```

## Notes

- Feature mismatch errors usually mean the residual table does not include the same feature columns used for training.
- The LIME pipeline uses one-hot encoded categoricals; align dummy columns before prediction.
- If explanations are noisy, increase `--n_samples` or ridge regularization.
