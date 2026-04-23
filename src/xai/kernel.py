from __future__ import annotations

"""
Distance and kernel computations for LIME.

This module defines functions to compute pairwise distances between a
set of perturbed instances and a reference instance, as well as to
compute exponential kernel weights based on those distances. The kernel
width can be set explicitly or determined via a median heuristic.
"""

import numpy as np


def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute row‑wise Euclidean distances between an array of points and a
    single reference point.

    Args:
        a: Array of shape (n_samples, n_features).
        b: Array of shape (n_features,) representing the reference point.

    Returns:
        Array of shape (n_samples,) containing distances.
    """
    return np.sqrt(np.sum((a - b[None, :]) ** 2, axis=1))


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute row‑wise cosine distance (1 - cosine similarity).

    Args:
        a: Array of shape (n_samples, n_features).
        b: Array of shape (n_features,) representing the reference point.

    Returns:
        Array of shape (n_samples,) containing distances.
    """
    a_norm = np.linalg.norm(a, axis=1) + 1e-12
    b_norm = np.linalg.norm(b) + 1e-12
    cos_sim = (a @ b) / (a_norm * b_norm)
    return 1.0 - cos_sim


def distances(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute distances between each row of 'a' and the vector 'b' using the
    specified metric.

    Args:
        a: Matrix of shape (n_samples, n_features).
        b: Vector of shape (n_features,).
        metric: 'euclidean' or 'cosine'.

    Returns:
        Array of shape (n_samples,) containing distances.
    """
    metric = metric.lower().strip()

    if metric == "euclidean":
        return _euclidean(a, b)
    

    if metric == "cosine":
        return _cosine(a, b)
    
    raise ValueError("metric must be 'euclidean' or 'cosine'")


def kernel_weights(d: np.ndarray, kernel_width: float | None) -> tuple[np.ndarray, float]:
    """
    Compute exponential kernel weights for distances.

    The weights are defined as exp(-d^2 / (kw^2)). If kernel_width is None,
    the width is set to the median of the distances. A small epsilon is
    added to avoid divide‑by‑zero.

    Args:
        d: Array of distances.
        kernel_width: Optional width of the kernel. If None, a heuristic
                      (median) is used.

    Returns:
        Tuple containing the array of weights and the kernel width used.
    """
    d = d.astype(float)

    if kernel_width is None:
        kw = float(np.median(d)) + 1e-12

    else:
        kw = float(kernel_width) + 1e-12

    w = np.exp(-(d ** 2) / (kw ** 2))
    return w, kw
