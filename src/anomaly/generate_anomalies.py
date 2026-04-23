from __future__ import annotations

"""
This module encapsulates the logic for filtering the residual table
according to the anomaly configuration and ranking the anomalies.
"""

import pandas as pd

from src.anomaly.thresholds import flag_anomalies
from src.utils.schema import TripSchema, AnomalyConfig


def generate_anomaly_table(
        residuals_df: pd.DataFrame,
        schema: TripSchema,
        cfg: AnomalyConfig,
    ) -> tuple[pd.DataFrame, dict]:

    """
    Create a sorted anomaly table based on residual values and configuration.

    Args:
        residuals_df: DataFrame with residuals. Must include columns for
                      trip_id, vehicle_id, target y, prediction yhat and residual.
        schema: Schema describing the relevant column names.
        cfg: Configuration controlling the thresholding method and parameters.

    Returns:
        anomalies_df: Subset of residuals_df containing only the anomalies.
        meta: Metadata about the threshold used and algorithm parameters.
    """
    # Ensure the required residual column is present
    if schema.residual not in residuals_df.columns:
        raise ValueError(f"Missing residual column: {schema.residual}")

    # Apply threshold to obtain a boolean mask of anomalies
    mask, meta = flag_anomalies(
        residuals=residuals_df[schema.residual],
        method=cfg.method,
        side=cfg.side,
        quantile=cfg.quantile,
        mad_z=cfg.mad_z,
    )

    anomalies = residuals_df.loc[mask].copy()

    # Rank anomalies: higher residual first for positive side,
    # by absolute residual for both sides
    if cfg.side == "positive":
        anomalies = anomalies.sort_values(schema.residual, ascending=False)
    else:
        anomalies = anomalies.assign(abs_residual=anomalies[schema.residual].abs())
        anomalies = anomalies.sort_values("abs_residual", ascending=False)

    anomalies["anomaly_rank"] = range(1, len(anomalies) + 1)
    return anomalies, meta
