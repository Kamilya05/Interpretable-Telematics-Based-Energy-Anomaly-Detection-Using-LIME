from __future__ import annotations

"""
LIME explanations for tabular data.

This module implements the core logic of the LIME algorithm for regression on
tabular data. It generates perturbed samples around a given instance,
computes kernel weights based on distances, fits a weighted linear surrogate
model and evaluates local fidelity. The results include the intercept,
feature weights, top contributions and fidelity metrics.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.xai.perturb import perturb_tabular
from src.xai.kernel import distances, kernel_weights
from src.xai.surrogate import weighted_ridge_closed_form, select_top_k
from src.xai.fidelity import weighted_r2, weighted_rmse


@dataclass(frozen=True)
class LimeExplanation:
    """
    Container for the result of a single LIME explanation.

    Attributes:
        trip_id: Identifier of the explained instance.
        intercept: Intercept of the surrogate model.
        weights: Dict mapping each feature (one-hot encoded where applicable)
                to its contribution at the reference point.
        top_features: List of (feature_name, contribution) tuples sorted
                      by importance.
        kernel_width: Width used in the kernel for weighting.
        local_r2: Weighted R^2 of the surrogate.
        local_rmse: Weighted RMSE of the surrogate.
    """
    trip_id: Any
    intercept: float
    weights: Dict[str, float]
    top_features: List[tuple[str, float]]
    kernel_width: float
    local_r2: float
    local_rmse: float


def _encode_for_distance(X: pd.DataFrame, x0: pd.Series) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Encode a DataFrame and a reference row into numeric arrays for distance computation.

    Numeric columns remain as they are. Categorical columns are one‑hot
    encoded using pandas.get_dummies. The union of categories between X
    and x0 is used so that x0 is encoded consistently.

    Args:
        X: DataFrame of shape (n_samples, n_features).
        x0: Series of shape (n_features,) representing the reference instance.

    Returns:
        Tuple (X_enc, x0_enc, feature_names) where:
          X_enc: Encoded feature matrix (n_samples, n_encoded_features)
          x0_enc: Encoded reference vector (n_encoded_features,)
          feature_names: Names of encoded columns.
    """
    df = X.copy()
    df0 = x0.to_frame().T.copy()
    union = pd.concat([df, df0], axis=0, ignore_index=True)
    union_enc = pd.get_dummies(union, drop_first=False)
    X_enc = union_enc.iloc[:-1].to_numpy(dtype=float)
    x0_enc = union_enc.iloc[-1].to_numpy(dtype=float)
    feature_names = list(union_enc.columns)
    return X_enc, x0_enc, feature_names


def explain_instance(
        trip_id: Any,
        x0: pd.Series,
        background_df: pd.DataFrame,
        feature_cols: List[str],
        black_box_predict: Callable[[pd.DataFrame], np.ndarray],
        n_samples: int = 5000,
        kernel_width: Optional[float] = None,
        distance_metric: str = "euclidean",
        ridge_alpha: float = 1.0,
        top_k: int = 10,
        random_state: int = 42,
    ) -> LimeExplanation:
    """
    Generate a LIME explanation for a given instance x0.

    Steps:
      1) Generate synthetic samples around x0 (Z).
      2) Obtain model predictions f(Z).
      3) Encode Z and x0 into numeric arrays for distance computation.
      4) Compute distances and kernel weights.
      5) Fit a weighted ridge regression (surrogate) on one‑hot features.
      6) Evaluate local fidelity (R^2 and RMSE).
      7) Select top features based on contribution.

    Args:
        trip_id: Identifier of the explained instance.
        x0: Series containing the features of the instance.
        background_df: DataFrame used for perturbation and categorical sampling.
        feature_cols: Names of the feature columns used by the model.
        black_box_predict: Function f(Z) returning predictions on a DataFrame.
        n_samples: Number of synthetic samples to generate.
        kernel_width: Width of the exponential kernel. If None, median heuristic.
        distance_metric: 'euclidean' or 'cosine'.
        ridge_alpha: Ridge regularization strength for the surrogate.
        top_k: Number of top features to include in the explanation.
        random_state: Seed for reproducibility.

    Returns:
        A LimeExplanation object containing all relevant information.
    """
    # 1) Generate perturbations around x0
    Z = perturb_tabular(
        x0=x0,
        background_df=background_df,
        feature_cols=feature_cols,
        n_samples=n_samples,
        random_state=random_state,
        noise_scale=1.0,
    )

    # 2) Evaluate black box model on perturbed samples
    y_f = black_box_predict(Z).astype(float)

    # 3) Encode for distance computation (one‑hot as needed)
    Z_enc, x0_enc, enc_names = _encode_for_distance(Z[feature_cols], x0[feature_cols])

    # 4) Distances and kernel weights
    d = distances(Z_enc, x0_enc, metric=distance_metric)
    w, kw_used = kernel_weights(d, kernel_width)

    # 5) Prepare design matrix for surrogate using one‑hot encoding
    X_sur = pd.get_dummies(Z[feature_cols], drop_first=False)
    x0_sur = pd.get_dummies(x0[feature_cols].to_frame().T, drop_first=False)
    # Align columns to ensure same shape
    X_sur, x0_sur = X_sur.align(x0_sur, join="outer", axis=1, fill_value=0.0)
    feature_names = list(X_sur.columns)

    beta, intercept = weighted_ridge_closed_form(
        X=X_sur.to_numpy(dtype=float),
        y=y_f,
        w=w,
        alpha=ridge_alpha,
    )

    # 6) Evaluate fidelity
    y_g = intercept + X_sur.to_numpy(dtype=float) @ beta
    r2 = weighted_r2(y_f, y_g, w)
    rmse = weighted_rmse(y_f, y_g, w)

    # 7) Compute contributions at x0
    x0_vec = x0_sur.to_numpy(dtype=float).reshape(-1)
    contrib = beta * x0_vec
    weights_dict = {feature_names[i]: float(contrib[i]) for i in range(len(feature_names))}
    top = select_top_k(contrib, feature_names, k=top_k)

    return LimeExplanation(
        trip_id=trip_id,
        intercept=float(intercept),
        weights=weights_dict,
        top_features=top,
        kernel_width=float(kw_used),
        local_r2=float(r2),
        local_rmse=float(rmse),
    )
