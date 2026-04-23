from __future__ import annotations

"""Command-line tool to generate a LIME explanation for a specific trip.

Recommended usage:
- scored trip table produced by the full notebook
- full training artifact with raw-feature metadata

This version is intentionally explicit because the original CLI could fail for
committed artifacts in two ways:
1. the saved `.joblib` file was a dictionary, not a bare model
2. `residuals.parquet` did not contain the raw features required for LIME
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.design import make_design_matrix, require_columns
from src.utils.artifact import build_prediction_fn, load_training_artifact
from src.utils.io import ensure_dir, read_table, save_json
from src.utils.schema import TripSchema
from src.viz.plots import plot_lime_bar
from src.xai.lime import explain_instance


def _resolve_feature_frame(df: pd.DataFrame, artifact, schema: TripSchema) -> tuple[pd.DataFrame, list[str]]:
    """Return the feature frame used for explanation.
    Why:
    - full XAI artifact => explain on raw feature columns
    - baseline artifact => explain on already encoded design columns if present
    """
    if artifact.raw_feature_spec is not None:
        raw_cols = artifact.raw_feature_spec.raw_features
        require_columns(df, raw_cols, context="raw-feature LIME explanation")
        return df[raw_cols].copy(), raw_cols

    require_columns(df, artifact.design_columns, context="design-matrix explanation")
    return df[artifact.design_columns].copy(), artifact.design_columns


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain a single trip using LIME.")

    ap.add_argument(
        "--scored_table_path",
        "--residuals_path",
        dest="scored_table_path",
        required=True,
        help=(
            "Path to the scored trip table. For the best experience, use the table "
            "produced by final_telematics_lime_project.ipynb because it includes raw features."
        ),
    )

    ap.add_argument(
        "--artifact_path",
        "--model_path",
        dest="artifact_path",
        required=True,
        help="Path to the saved training artifact (.joblib)",
    )
    
    ap.add_argument("--trip_id", required=True, help="Identifier of the trip to explain")
    ap.add_argument("--out_dir", default="outputs/explanations", help="Directory to save explanation artifacts")
    ap.add_argument("--n_samples", type=int, default=5000, help="Number of perturbation samples")
    ap.add_argument("--top_k", type=int, default=10, help="Number of top features to report")
    ap.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge regularization strength for the surrogate")
    ap.add_argument("--distance_metric", default="euclidean", choices=["euclidean", "cosine"], help="Distance metric")
    ap.add_argument("--kernel_width", type=float, default=None, help="Kernel width for the exponential kernel")
    ap.add_argument("--random_state", type=int, default=42, help="Seed for reproducibility")
    args = ap.parse_args()

    schema = TripSchema()
    out_dir = ensure_dir(args.out_dir)

    df = read_table(args.scored_table_path)
    artifact = load_training_artifact(args.artifact_path)
    predict_fn = build_prediction_fn(artifact)

    feature_df, feature_cols = _resolve_feature_frame(df, artifact, schema)

    trip_rows = df[df[schema.trip_id].astype(str) == str(args.trip_id)]
    if len(trip_rows) != 1:
        raise ValueError(f"Expected exactly 1 row for trip_id={args.trip_id}, got {len(trip_rows)}")

    x0_row = trip_rows.iloc[0]
    x0 = feature_df.loc[x0_row.name]

    if artifact.raw_feature_spec is not None:
        if artifact.background_raw is None:
            raise ValueError("Artifact is missing background_raw required for raw-feature LIME explanations.")
        background_df = artifact.background_raw.copy().reset_index(drop=True)
    else:
        background_df = feature_df.copy().reset_index(drop=True)

    exp = explain_instance(
        trip_id=args.trip_id,
        x0=x0,
        background_df=background_df,
        feature_cols=feature_cols,
        black_box_predict=predict_fn,
        n_samples=args.n_samples,
        kernel_width=args.kernel_width,
        distance_metric=args.distance_metric,
        ridge_alpha=args.ridge_alpha,
        top_k=args.top_k,
        random_state=args.random_state,
    )

    baseline_prediction = float(np.asarray(predict_fn(pd.DataFrame([x0])), dtype=float)[0])
    actual = float(x0_row[schema.y]) if schema.y in x0_row.index else None
    residual = float(x0_row[schema.residual]) if schema.residual in x0_row.index else None

    exp_path = out_dir / f"trip_{args.trip_id}_lime.json"
    save_json(
        {
            "trip_id": exp.trip_id,
            "artifact_type": artifact.metadata.get("artifact_type"),
            "baseline_prediction": baseline_prediction,
            "actual_energy_per_km": actual,
            "residual": residual,
            "intercept": exp.intercept,
            "kernel_width": exp.kernel_width,
            "local_r2": exp.local_r2,
            "local_rmse": exp.local_rmse,
            "top_features": exp.top_features,
            "weights": exp.weights,
        },
        exp_path,
    )

    plot_path = out_dir / f"trip_{args.trip_id}_lime_bar.png"
    plot_lime_bar(exp.top_features, plot_path, title=f"LIME contributions for trip {args.trip_id}")

    print(f"Saved explanation: {exp_path}")
    print(f"Saved bar plot: {plot_path}")
    print(f"local_r2={exp.local_r2:.3f}, local_rmse={exp.local_rmse:.3f}")


if __name__ == "__main__":
    main()
