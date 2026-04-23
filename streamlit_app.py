from __future__ import annotations

import gzip
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai.lime import explain_instance
from src.xai.shap import ManualSHAP

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACT_PATH = PROJECT_ROOT / "regression_model/models/xgb_energy_artifact.joblib"
OUTPUT_ZIP_PATH = PROJECT_ROOT / "notebooks/baseline_implementation/output.zip"
SHAP_DIR = PROJECT_ROOT / "outputs_blogpost/xai/shap"
LIME_DIR = PROJECT_ROOT / "outputs_blogpost/xai/lime"
TABLE_DIR = PROJECT_ROOT / "outputs_blogpost/tables"

# ── Feature config ────────────────────────────────────────────────────────────
_RAW_NUMERIC = [
    "duration_min", "distance_km", "speed_mean", "speed_var",
    "accel_mean", "accel_var", "accel_p95", "stop_go_ratio",
    "idle_time_min", "fuel_energy_kWh", "battery_energy_kWh",
    "ac_energy_kWh", "heater_energy_kWh", "hv_current_abs_mean",
    "hv_current_abs_p95", "hv_voltage_mean", "maf_mean", "maf_p95",
    "Generalized_Weight",
]
_TARGET_COMPONENTS = {"fuel_energy_kWh", "battery_energy_kWh", "ac_energy_kWh", "heater_energy_kWh"}
MODEL_NUMERIC = [c for c in _RAW_NUMERIC if c not in _TARGET_COMPONENTS]
CAT_FEATURES = ["VehicleType", "Vehicle Class", "Transmission", "Drive Wheels"]

# ── Design matrix ─────────────────────────────────────────────────────────────
def _make_design(df: pd.DataFrame, feature_columns: list, feature_medians: dict) -> pd.DataFrame:
    raw = df.reindex(columns=MODEL_NUMERIC + CAT_FEATURES).copy()
    for c in MODEL_NUMERIC:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(feature_medians.get(c, 0.0))
    for c in CAT_FEATURES:
        raw[c] = raw[c].astype("string").fillna("NO DATA")
    X = pd.get_dummies(raw, columns=CAT_FEATURES, dummy_na=False)
    return X.reindex(columns=feature_columns, fill_value=0.0).astype(float)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_everything() -> dict:
    art = joblib.load(ARTIFACT_PATH)
    with zipfile.ZipFile(OUTPUT_ZIP_PATH) as zf:
        raw_gz = zf.read("outputs_final/cache/trip_table_scored.csv.gz")
    df = pd.read_csv(BytesIO(gzip.decompress(raw_gz)))
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    X_test = _make_design(test_df, art["feature_columns"], art["feature_medians"])
    anomaly_df = (
        test_df[test_df["is_anomaly"]]
        .sort_values("residual", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "model": art["model"],
        "feature_columns": art["feature_columns"],
        "feature_medians": art["feature_medians"],
        "background": art["lime_background"].copy(),
        "test_df": test_df,
        "X_test": X_test,
        "anomaly_df": anomaly_df,
    }

@st.cache_resource
def get_shap_explainer() -> ManualSHAP:
    d = load_everything()
    return ManualSHAP(
        model=d["model"],
        background=d["X_test"],
        random_state=42,
        max_background_rows=64,
    )

@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    p = TABLE_DIR / "predictive_summary.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_precomputed_fidelity() -> dict:
    p = TABLE_DIR / "lime_case_summary.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {
        str(row["trip_id"]): {"local_r2": row["local_r2"], "local_rmse": row["local_rmse"]}
        for _, row in df.iterrows()
    }

# ── Precomputed CSV helpers ───────────────────────────────────────────────────
def _precomp_shap(trip_id: str) -> pd.DataFrame | None:
    p = SHAP_DIR / f"trip_{trip_id}_shap.csv"
    return pd.read_csv(p) if p.exists() else None

def _precomp_lime(trip_id: str) -> tuple[pd.DataFrame | None, dict]:
    p = LIME_DIR / f"trip_{trip_id}_lime.csv"
    if not p.exists():
        return None, {}
    return pd.read_csv(p), load_precomputed_fidelity().get(trip_id, {})

# ── Explanation resolution ────────────────────────────────────────────────────
def resolve(trip_id: str, method: str) -> tuple[pd.DataFrame | None, dict, str | None]:
    """Return (df, meta, source) where source is 'live', 'cached', or None."""
    live = st.session_state.live_results.get(trip_id, {})
    if method == "shap" and "shap_df" in live:
        return live["shap_df"], live.get("shap_params", {}), "live"
    if method == "lime" and "lime_df" in live:
        return live["lime_df"], live.get("lime_meta", {}), "live"
    if method == "shap":
        df = _precomp_shap(trip_id)
        return (df, {}, "cached") if df is not None else (None, {}, None)
    if method == "lime":
        df, meta = _precomp_lime(trip_id)
        return (df, meta, "cached") if df is not None else (None, {}, None)
    return None, {}, None

# ── Live computation ───────────────────────────────────────────────────────────
def _get_x0(trip_id: str) -> pd.Series:
    d = load_everything()
    idx = d["test_df"].index[d["test_df"]["trip_id"].astype(str) == str(trip_id)]
    if len(idx) == 0:
        raise ValueError(f"Trip {trip_id} not found in test set")
    return d["X_test"].loc[int(idx[0])]

def live_shap(trip_id: str, n_perms: int, top_k: int) -> pd.DataFrame:
    x0 = _get_x0(trip_id)
    phi = get_shap_explainer().permutation_shap_row(
        x0=x0, top_features=top_k, n_permutations=n_perms
    )
    return pd.DataFrame({"feature": phi.index, "contribution": phi.values})

def live_lime(trip_id: str, n_samples: int, ridge_alpha: float) -> tuple[pd.DataFrame, dict]:
    d = load_everything()
    x0 = _get_x0(trip_id)

    def predict(X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            d["model"].predict(X.reindex(columns=d["feature_columns"], fill_value=0.0))
        )

    exp = explain_instance(
        trip_id=trip_id,
        x0=x0,
        background_df=d["background"],
        feature_cols=d["feature_columns"],
        black_box_predict=predict,
        n_samples=n_samples,
        kernel_width=0.75 * np.sqrt(len(d["feature_columns"])),
        ridge_alpha=ridge_alpha,
        top_k=10,
        random_state=42,
    )
    df = pd.DataFrame(exp.top_features, columns=["feature", "contribution"])
    meta = {
        "local_r2": exp.local_r2,
        "local_rmse": exp.local_rmse,
        "intercept": exp.intercept,
        "kernel_width": exp.kernel_width,
    }
    return df, meta

# ── Chart builders ─────────────────────────────────────────────────────────────
def _bar_chart(df: pd.DataFrame, title: str) -> go.Figure:
    plot = (
        df.assign(_abs=lambda d: d["contribution"].abs())
        .sort_values("_abs", ascending=True)
        .tail(10)
    )
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in plot["contribution"]]
    fig = go.Figure(go.Bar(
        x=plot["contribution"],
        y=plot["feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.5f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=380,
        margin=dict(l=10, r=10, t=42, b=0),
        xaxis_title="Contribution to predicted energy_per_km",
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(automargin=True),
        showlegend=False,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def _comparison_chart(shap_df: pd.DataFrame, lime_df: pd.DataFrame, trip_id: str) -> go.Figure:
    s = shap_df.set_index("feature")["contribution"]
    l = lime_df.set_index("feature")["contribution"]
    merged = pd.DataFrame({"SHAP": s, "LIME": l}).fillna(0.0)
    merged["_rank"] = merged["SHAP"].abs() + merged["LIME"].abs()
    merged = merged.sort_values("_rank", ascending=True).tail(12)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Manual SHAP", y=merged.index, x=merged["SHAP"],
        orientation="h", marker_color="#8b5cf6",
        hovertemplate="%{y} SHAP: %{x:.5f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="LIME", y=merged.index, x=merged["LIME"],
        orientation="h", marker_color="#f59e0b",
        hovertemplate="%{y} LIME: %{x:.5f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(
        barmode="group",
        title=dict(text=f"trip {trip_id}", font=dict(size=14)),
        height=520,
        margin=dict(l=10, r=20, t=50, b=40),
        xaxis_title="Contribution",
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── App ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Anomaly Explainer",
    layout="wide",
)

st.markdown("""
<style>
/* Hide footer and top-right menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Typography — !important needed because Streamlit's stylesheet loads after */
html, body, h1, h2, h3, h4, h5, h6, p, span, div, label, button {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] p {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
}
[data-testid="stMetricValue"] {
    font-size: 22px;
    font-weight: 600;
    color: #0f172a;
}

/* Sidebar border */
[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.title("Vehicle Energy Anomaly Explainer")
st.caption("Telematics-based anomaly detection — Manual SHAP and LIME explainability")

with st.spinner("Loading model and data…"):
    data = load_everything()

anomaly_df = data["anomaly_df"]
trip_ids = anomaly_df["trip_id"].astype(str).tolist()

# Session state
if "selected_trip" not in st.session_state:
    st.session_state.selected_trip = trip_ids[0]
if "live_results" not in st.session_state:
    st.session_state.live_results = {}
if "initialized" not in st.session_state:
    st.session_state.initialized = set()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Trip")

    def _trip_label(t: str) -> str:
        row = anomaly_df[anomaly_df["trip_id"].astype(str) == t].iloc[0]
        return f"{t}  (residual {row['residual']:.3f})"

    selected = st.selectbox(
        "Anomalous trip",
        trip_ids,
        index=trip_ids.index(st.session_state.selected_trip),
        format_func=_trip_label,
    )
    st.session_state.selected_trip = selected

    st.divider()
    st.subheader("SHAP parameters")
    n_perms = st.slider("Permutations", 20, 300, 100, step=10,
        help="Higher = more accurate, but slower. 100 takes ~15–30 s.")
    shap_top_k = st.slider("Top features", 5, 20, 12,
        help="Number of features (by model importance) to compute SHAP for.")

    st.subheader("LIME parameters")
    n_samples = st.slider("Samples", 500, 6000, 2000, step=500,
        help="Perturbed samples used to fit the local surrogate.")
    ridge_alpha = st.slider("Ridge α", 0.01, 2.0, 0.10, step=0.01,
        help="Regularisation strength for the LIME surrogate.")

    st.divider()
    rerun_clicked = st.button("Re-run live", type="primary", use_container_width=True,
        help="Force-recompute both methods with the current parameters.")

trip_id = st.session_state.selected_trip

# ── Compute explanations ───────────────────────────────────────────────────────
if rerun_clicked:
    with st.spinner("Running Manual SHAP…"):
        _shap_df = live_shap(trip_id, n_perms, shap_top_k)
    with st.spinner("Running LIME…"):
        _lime_df, _lime_meta = live_lime(trip_id, n_samples, ridge_alpha)
    st.session_state.live_results[trip_id] = {
        "shap_df": _shap_df, "shap_params": {"n_permutations": n_perms, "top_features": shap_top_k},
        "lime_df": _lime_df, "lime_meta": _lime_meta,
        "lime_params": {"n_samples": n_samples, "ridge_alpha": ridge_alpha},
    }
    st.session_state.initialized.add(trip_id)

elif trip_id not in st.session_state.initialized:
    # Auto-run only what's missing (no precomputed file)
    need_shap = _precomp_shap(trip_id) is None
    need_lime = _precomp_lime(trip_id)[0] is None
    if need_shap or need_lime:
        label = " and ".join(filter(None, ["SHAP" if need_shap else "", "LIME" if need_lime else ""]))
        with st.spinner(f"No cache for trip {trip_id} — computing {label}…"):
            result = st.session_state.live_results.get(trip_id, {})
            if need_shap:
                result["shap_df"] = live_shap(trip_id, n_perms, shap_top_k)
                result["shap_params"] = {"n_permutations": n_perms, "top_features": shap_top_k}
            if need_lime:
                _lime_df, _lime_meta = live_lime(trip_id, n_samples, ridge_alpha)
                result["lime_df"] = _lime_df
                result["lime_meta"] = _lime_meta
                result["lime_params"] = {"n_samples": n_samples, "ridge_alpha": ridge_alpha}
            st.session_state.live_results[trip_id] = result
    st.session_state.initialized.add(trip_id)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_explorer, tab_compare = st.tabs(["Overview", "Trip Explorer", "Method Comparison"])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab_overview:
    metrics = load_metrics()
    if metrics is not None:
        row = metrics[metrics["split"] == "test"].iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("MAE", f"{row['mae']:.4f}")
        c2.metric("RMSE", f"{row['rmse']:.4f}")
        c3.metric("R²", f"{row['r2']:.4f}")
        c4.metric("Test trips", int(row["n_trips"]))
        c5.metric("Anomalies", int(row["n_anomalies"]))
        c6.metric("Anomaly rate", f"{row['anomaly_rate']:.1%}")
        st.divider()

    st.subheader("Flagged anomalies — test split")
    st.caption("Click a row to select a trip, then switch to Trip Explorer.")

    display_cols = [c for c in [
        "trip_id", "VehId", "energy_per_km", "predicted_energy_per_km",
        "residual", "VehicleType", "Transmission", "Drive Wheels",
    ] if c in anomaly_df.columns]

    tbl = anomaly_df[display_cols].copy()
    for col in ["energy_per_km", "predicted_energy_per_km", "residual"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].round(4)

    sel = st.dataframe(
        tbl,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        height=580,
    )
    if sel.selection.rows:
        new_id = str(tbl.iloc[sel.selection.rows[0]]["trip_id"])
        if new_id != st.session_state.selected_trip:
            st.session_state.selected_trip = new_id
            st.rerun()

# ── Tab 2: Trip Explorer ───────────────────────────────────────────────────────
with tab_explorer:
    trip_row = anomaly_df[anomaly_df["trip_id"].astype(str) == trip_id].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Actual energy/km", f"{trip_row['energy_per_km']:.4f}")
    c2.metric("Predicted", f"{trip_row['predicted_energy_per_km']:.4f}")
    c3.metric("Residual (anomaly score)", f"{trip_row['residual']:.4f}")
    c4.metric("Vehicle type", str(trip_row.get("VehicleType", "—")))
    c5.metric("Transmission", str(trip_row.get("Transmission", "—")))

    st.divider()

    shap_df, shap_params, shap_src = resolve(trip_id, "shap")
    lime_df, lime_meta, lime_src = resolve(trip_id, "lime")

    col_shap, col_lime = st.columns(2)

    with col_shap:
        badge = "(live)" if shap_src == "live" else "(cached)"
        st.subheader(f"Manual SHAP   {badge}")

        if shap_src == "live" and shap_params:
            st.caption(
                f"permutations: {shap_params.get('n_permutations')} · "
                f"top features: {shap_params.get('top_features')}"
            )
        elif shap_src == "cached":
            st.caption("Loaded from precomputed output.")

        if shap_df is not None:
            st.plotly_chart(_bar_chart(shap_df, f"SHAP contributions — trip {trip_id}"),
                            use_container_width=True)

            explainer = get_shap_explainer()
            base = explainer.base_value()
            recon = base + float(shap_df["contribution"].sum())
            predicted = float(trip_row["predicted_energy_per_km"])

            mc1, mc2 = st.columns(2)
            mc1.metric("Base value  E[ŷ]", f"{base:.4f}",
                       help="Average prediction over the background set — the SHAP reference point.")
            mc2.metric(
                "Reconstruction",
                f"{recon:.4f}",
                delta=f"{recon - predicted:+.4f} vs ŷ",
                delta_color="off",
                help="base value + Σ shown ϕᵢ. Gap is non-zero because only a feature subset is shown.",
            )
            with st.expander("About the reconstruction gap"):
                st.write(
                    "SHAP is additive: base_value + Σ(all ϕᵢ) = ŷ exactly. "
                    "Because only the top-K features by model importance are computed here, "
                    "the partial sum won't exactly match ŷ. "
                    "Increase **Top features** in the sidebar or click **Re-run live** to reduce the gap."
                )
        else:
            st.info("No explanation available — click **Re-run live** in the sidebar.")

    with col_lime:
        badge = "(live)" if lime_src == "live" else "(cached)"
        st.subheader(f"LIME   {badge}")

        if lime_src == "live":
            live_entry = st.session_state.live_results.get(trip_id, {})
            p = live_entry.get("lime_params", {})
            if p:
                st.caption(f"samples: {p.get('n_samples')} · ridge α: {p.get('ridge_alpha')}")
        elif lime_src == "cached":
            st.caption("Loaded from precomputed output.")

        if lime_df is not None:
            st.plotly_chart(_bar_chart(lime_df, f"LIME contributions — trip {trip_id}"),
                            use_container_width=True)

            r2_val = lime_meta.get("local_r2")
            rmse_val = lime_meta.get("local_rmse")

            lc1, lc2 = st.columns(2)
            lc1.metric(
                "Local R²",
                f"{r2_val:.4f}" if r2_val is not None else "—",
                help="Weighted R² of the linear surrogate in the local neighbourhood. Higher = more faithful.",
            )
            lc2.metric(
                "Local RMSE",
                f"{rmse_val:.5f}" if rmse_val is not None else "—",
                help="Weighted RMSE of the surrogate predictions vs the black-box on perturbed samples.",
            )

            details = []
            if lime_meta.get("intercept") is not None:
                details.append(f"intercept: {lime_meta['intercept']:.4f}")
            if lime_meta.get("kernel_width") is not None:
                details.append(f"kernel width: {lime_meta['kernel_width']:.3f}")
            if details:
                st.caption(" · ".join(details))

            if r2_val is not None and r2_val < 0.4:
                st.warning(
                    "Local R² is low — the linear surrogate may not faithfully approximate the "
                    "black-box model around this trip. Treat these feature attributions with caution."
                )
        else:
            st.info("No explanation available — click **Re-run live** in the sidebar.")

# ── Tab 3: Method Comparison ───────────────────────────────────────────────────
with tab_compare:
    shap_df, _, shap_src = resolve(trip_id, "shap")
    lime_df, lime_meta, lime_src = resolve(trip_id, "lime")

    if shap_df is None or lime_df is None:
        missing = []
        if shap_df is None:
            missing.append("SHAP")
        if lime_df is None:
            missing.append("LIME")
        st.info(
            f"Missing {' and '.join(missing)} for trip {trip_id}. "
            "Click **Re-run live** in the sidebar to compute both."
        )
    else:
        st.subheader(f"SHAP vs LIME")
        st.plotly_chart(_comparison_chart(shap_df, lime_df, trip_id), use_container_width=True)

        st.divider()
        col_stats, col_sets = st.columns(2)

        shap_top = set(
            shap_df.assign(_a=lambda d: d["contribution"].abs()).nlargest(10, "_a")["feature"]
        )
        lime_top = set(
            lime_df.assign(_a=lambda d: d["contribution"].abs()).nlargest(10, "_a")["feature"]
        )
        common = shap_top & lime_top

        with col_stats:
            st.subheader("Agreement metrics")
            st.metric("Features in common (top 10 each)", len(common))
            st.metric("SHAP-only features", len(shap_top - lime_top))
            st.metric("LIME-only features", len(lime_top - shap_top))

            all_f = list(shap_top | lime_top)
            sd = dict(zip(shap_df["feature"], shap_df["contribution"].abs()))
            ld = dict(zip(lime_df["feature"], lime_df["contribution"].abs()))
            sv = [sd.get(f, 0.0) for f in all_f]
            lv = [ld.get(f, 0.0) for f in all_f]

            if len(all_f) >= 3:
                try:
                    from scipy.stats import spearmanr
                    rho, pval = spearmanr(sv, lv)
                    st.metric(
                        "Spearman ρ (importance ranks)",
                        f"{rho:.3f}",
                        help=f"Rank correlation of |contribution| across features. p = {pval:.3f}",
                    )
                except ImportError:
                    st.caption("Install scipy to see rank correlation.")

            if lime_meta.get("local_r2") is not None:
                st.divider()
                r2 = lime_meta["local_r2"]
                quality = "excellent" if r2 > 0.8 else "moderate" if r2 > 0.4 else "poor"
                st.markdown("**LIME surrogate fidelity**")
                st.write(
                    f"Local R² = **{r2:.4f}** ({quality}). "
                    + (
                        "The surrogate faithfully approximates the model locally — "
                        "LIME attributions are reliable for this trip."
                        if quality != "poor"
                        else "The surrogate fit is weak. LIME attributions may not reflect the true "
                             "model behaviour around this trip."
                    )
                )

        with col_sets:
            st.subheader("Feature sets")
            if common:
                st.markdown("**In both top-10 lists**")
                for f in sorted(common):
                    shap_v = sd.get(f, 0.0)
                    lime_v = ld.get(f, 0.0)
                    st.write(f"• {f}  —  SHAP |ϕ| {shap_v:.5f} / LIME |ϕ| {lime_v:.5f}")
            if shap_top - lime_top:
                st.markdown("**SHAP only**")
                st.write(", ".join(sorted(shap_top - lime_top)))
            if lime_top - shap_top:
                st.markdown("**LIME only**")
                st.write(", ".join(sorted(lime_top - shap_top)))
