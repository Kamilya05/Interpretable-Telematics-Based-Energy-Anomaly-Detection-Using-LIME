from __future__ import annotations

"""
Anomaly thresholding utilities.

This module provides functions to compute thresholds for residuals and to
determine which trips are anomalous. Two methods are supported: quantile
thresholding and MAD‑z (robust z‑score) thresholding. Each method can
operate on the positive residuals alone or both sides of the residual
distribution.
"""

import numpy as np
import pandas as pd


def quantile_threshold(residuals: pd.Series, q: float) -> float:
    """
    Compute the quantile threshold of the residuals.

    Args:
        residuals: Series of residual values.
        q: Quantile in the (0, 1) range. For positive anomalies, q should
           represent the fraction of observations considered anomalous.

    Returns:
        Threshold value such that residual >= threshold corresponds to the
        top‑q fraction of residuals.
    """

    if not 0.0 < q < 1.0:
        raise ValueError("q must be between 0 and 1")
    
    return float(np.nanquantile(residuals.values, q))


def mad_threshold(residuals: pd.Series, z: float) -> tuple[float, float, float]:
    """
    Compute a robust z‑score threshold using the median and median absolute
    deviation (MAD).

    Args:
        residuals: Series of residuals.
        z: Threshold on the robust z‑score (e.g., 3.5)

    Returns:
        A tuple (median, mad, threshold) where threshold = median + z * mad / 0.6745
        for the positive side. 0.6745 converts the MAD to be comparable to the
        standard deviation for a normal distribution.
    """

    x = residuals.values.astype(float)
    median = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - median))) + 1e-12
    thr = median + (z * mad / 0.6745)
    return median, mad, float(thr)


def flag_anomalies(
        residuals: pd.Series,
        method: str,
        side: str = "positive",
        quantile: float = 0.98,
        mad_z: float = 3.5,
    ) -> tuple[pd.Series, dict]:

    """
    Determine anomalies based on residuals.

    Args:
        residuals: Series of residual values.
        method: 'quantile' or 'mad_z'. Determines how to compute the threshold.
        side: 'positive' or 'both'. If 'positive', only positive residuals are
              considered anomalies. If 'both', absolute residuals are used.
        quantile: Quantile for quantile thresholding.
        mad_z: Robust z‑score threshold for MAD thresholding.

    Returns:
        A boolean mask indicating which rows are anomalous and a dictionary of
        metadata describing the threshold(s) used.
    """

    method = method.lower().strip()
    side = side.lower().strip()
    
    if side not in {"positive", "both"}:
        raise ValueError("side must be 'positive' or 'both'")

    meta: dict = {"method": method, "side": side}

    if method == "quantile":
        thr_pos = quantile_threshold(residuals, quantile)
        meta.update({"quantile": quantile, "threshold_pos": thr_pos})
        if side == "positive":
            return (residuals >= thr_pos), meta
        # For both sides, compute threshold on absolute residuals.
        thr_abs = float(np.nanquantile(np.abs(residuals.values), quantile))
        meta.update({"threshold_abs": thr_abs})
        return (np.abs(residuals) >= thr_abs), meta

    if method == "mad_z":
        median, mad, thr_pos = mad_threshold(residuals, mad_z)
        meta.update({"mad_z": mad_z, "median": median, "mad": mad, "threshold_pos": thr_pos})
        if side == "positive":
            return (residuals >= thr_pos), meta
        # For both sides, compute robust z‑scores on absolute residuals.
        x = residuals.values.astype(float)
        robust_z = 0.6745 * np.abs(x - median) / (mad + 1e-12)
        meta.update({"threshold_robust_z": mad_z})
        return (robust_z >= mad_z), meta

    raise ValueError("method must be 'quantile' or 'mad_z'")
