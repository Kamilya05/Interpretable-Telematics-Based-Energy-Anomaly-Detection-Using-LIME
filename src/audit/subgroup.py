from __future__ import annotations
import math
import pandas as pd

def subgroup_regression_metrics(df: pd.DataFrame, group_col: str, y_true_col: str, y_pred_col: str, anomaly_col: str | None = None) -> pd.DataFrame:
    rows = []

    for value, part in df.groupby(group_col, dropna=False):
        err = part[y_true_col] - part[y_pred_col]
        ss_res = float(((part[y_true_col] - part[y_pred_col]) ** 2).sum())
        ss_tot = float(((part[y_true_col] - part[y_true_col].mean()) ** 2).sum())
        rows.append({
            "group_col": group_col,
            "group_value": value,
            "n": int(len(part)),
            "mae": float(err.abs().mean()),
            "rmse": float(math.sqrt((err ** 2).mean())),
            "r2": float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot,
            "mean_residual": float(err.mean()),
            "anomaly_rate": float(part[anomaly_col].mean()) if anomaly_col and anomaly_col in part.columns else float("nan"),
        })
        
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
