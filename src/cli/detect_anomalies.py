from __future__ import annotations

"""
Command‑line interface to detect anomalies based on residuals.

This script reads a residuals file (Parquet or CSV), applies a thresholding
method to identify anomalous trips, writes the resulting anomaly table and
saves the threshold metadata to disk. It depends on the functions in
src.anomaly.generate_anomalies and the TripSchema/AnomalyConfig dataclasses.
"""

import argparse

from src.utils.io import read_table, write_table, save_json
from src.utils.schema import TripSchema, AnomalyConfig
from src.anomaly.generate_anomalies import generate_anomaly_table


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect anomalies based on residuals.")

    ap.add_argument(
        "--residuals_path",
        required=True,
        help="Path to residuals table (e.g. outputs/residuals.parquet)",
    )

    ap.add_argument(
        "--out_anomalies_path",
        default="outputs/anomalies.parquet",
        help="Where to write the anomalies table",
    )

    ap.add_argument(
        "--out_meta_path",
        default="outputs/anomaly_config.json",
        help="Where to write the anomaly threshold metadata",
    )

    ap.add_argument(
        "--method",
        default="quantile",
        choices=["quantile", "mad_z"],
        help="Thresholding method",
    )

    ap.add_argument(
        "--side",
        default="positive",
        choices=["positive", "both"],
        help="Which side of residual distribution to consider anomalous",
    )

    ap.add_argument(
        "--quantile",
        type=float,
        default=0.98,
        help="Quantile for quantile thresholding",
    )

    ap.add_argument(
        "--mad_z",
        type=float,
        default=3.5,
        help="Robust z-score threshold for MAD-based thresholding",
    )

    args = ap.parse_args()

    # Read residuals table
    df = read_table(args.residuals_path)

    schema = TripSchema()
    
    cfg = AnomalyConfig(
        method=args.method,
        side=args.side,
        quantile=args.quantile,
        mad_z=args.mad_z,
    )

    anomalies, meta = generate_anomaly_table(df, schema, cfg)

    write_table(anomalies, args.out_anomalies_path)
    save_json(meta, args.out_meta_path)

    print(f"Saved anomalies: {args.out_anomalies_path} (n={len(anomalies)})")
    print(f"Saved config: {args.out_meta_path}")


if __name__ == "__main__":
    main()
