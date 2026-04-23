from __future__ import annotations

"""Helpers for turning raw trip-level telematics features into the design matrix
used by the XGBoost regressor.

Why this module exists:
- the baseline repository stored more than one artifact format
- LIME explanations need access to *raw* trip features, not only residuals
- preprocessing must exactly match the model's training-time encoding

This module centralizes that contract.
"""

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RawFeatureSpec:
    numeric_features: list[str]
    categorical_features: list[str]

    @property
    def raw_features(self) -> list[str]:
        return self.numeric_features + self.categorical_features


def build_feature_spec_from_artifact(artifact: dict) -> RawFeatureSpec:
    """Create a raw-feature specification from a saved training artifact."""
    return RawFeatureSpec(
        numeric_features=list(artifact.get("numeric_feature_columns", [])),
        categorical_features=list(artifact.get("categorical_feature_columns", [])),
    )


def fill_raw_features(
        df: pd.DataFrame,
        spec: RawFeatureSpec,
        numeric_fill: dict[str, float] | None = None,
    ) -> pd.DataFrame:

    """Fill missing values in raw features using the same logic as training.
    Numeric columns are coerced to float and filled with medians.
    Categorical columns are converted to strings and filled with ``NO DATA``.
    """
    out = df.copy()

    if numeric_fill is None:
        numeric_fill = {
            col: float(pd.to_numeric(out[col], errors="coerce").median())
            for col in spec.numeric_features
        }

    for col in spec.numeric_features:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(numeric_fill[col])

    for col in spec.categorical_features:
        out[col] = out[col].astype("string").fillna("NO DATA")

    return out


def make_design_matrix(
        df: pd.DataFrame,
        spec: RawFeatureSpec,
        design_columns: list[str] | None = None,
        numeric_fill: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    
    """Convert raw trip rows into the model design matrix.

    This mirrors the logic used in the final notebook:
    - fill raw numeric/categorical values
    - one-hot encode categorical columns
    - align to the saved design columns
    """
    raw = fill_raw_features(df[spec.raw_features], spec=spec, numeric_fill=numeric_fill)
    X = pd.get_dummies(raw, columns=spec.categorical_features, dummy_na=False)

    if design_columns is None:
        design_columns = list(X.columns)
    X = X.reindex(columns=design_columns, fill_value=0.0)

    if numeric_fill is None:
        numeric_fill = {col: float(raw[col].median()) for col in spec.numeric_features}

    return X.astype(float), numeric_fill, list(design_columns)


def require_columns(df: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    """Raise a clear error if the required columns are missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}. "
            "Make sure you are using the scored trip table produced by the full notebook."
        )
