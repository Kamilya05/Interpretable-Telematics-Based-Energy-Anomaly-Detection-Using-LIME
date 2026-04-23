from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ManualSHAP:
    """
    Shapley-based explainer implemented from first principles (no shap library).

    Supports exact Shapley values (small feature subsets) and permutation-based
    approximation for practical use on all features.
    """

    def __init__(
        self,
        model,
        background: pd.DataFrame,
        feature_names: Optional[list[str]] = None,
        random_state: int = 42,
        max_background_rows: Optional[int] = 64,
    ):
        if not isinstance(background, pd.DataFrame):
            raise TypeError("background must be a pandas DataFrame")
        if background.empty:
            raise ValueError("background must not be empty")

        self.model = model
        self.random_state = random_state

        if max_background_rows is not None and len(background) > max_background_rows:
            self.background = background.sample(
                max_background_rows, random_state=random_state
            ).reset_index(drop=True)
        else:
            self.background = background.reset_index(drop=True).copy()

        self.feature_names = feature_names or list(self.background.columns)

        missing = [c for c in self.feature_names if c not in self.background.columns]
        if missing:
            raise ValueError(f"Missing features in background: {missing}")

        self.background = self.background[self.feature_names].copy()
        self.feature_to_idx = {f: i for i, f in enumerate(self.feature_names)}

        if hasattr(model, "feature_importances_"):
            self.feature_importance_series = pd.Series(
                model.feature_importances_, index=self.feature_names
            ).sort_values(ascending=False)
        else:
            self.feature_importance_series = pd.Series(
                np.ones(len(self.feature_names), dtype=float), index=self.feature_names
            ).sort_values(ascending=False)

        self._base_value_cached = float(
            np.asarray(self.model.predict(self.background)).reshape(-1).mean()
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_series(self, x0: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(x0, pd.DataFrame):
            if len(x0) != 1:
                raise ValueError("If x0 is a DataFrame it must contain exactly one row.")
            x0 = x0.iloc[0]
        if not isinstance(x0, pd.Series):
            raise TypeError("x0 must be a pandas Series or single-row DataFrame")
        missing = [c for c in self.feature_names if c not in x0.index]
        if missing:
            raise ValueError(f"x0 is missing required features: {missing}")
        return x0[self.feature_names].copy()

    def _predict_df(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X)).reshape(-1)

    def _coalition_value(self, x0: pd.Series, coalition: set[str]) -> float:
        X_masked = self.background.copy()
        for col in coalition:
            X_masked[col] = x0[col]
        return float(self._predict_df(X_masked).mean())

    def _resolve_feature_subset(
        self,
        features: Optional[list[str]] = None,
        top_features: Optional[int] = None,
    ) -> list[str]:
        if features is not None:
            for f in features:
                if f not in self.feature_names:
                    raise ValueError(f"Unknown feature: {f}")
            return list(features)
        if top_features is not None:
            return list(self.feature_importance_series.head(top_features).index)
        return list(self.feature_names)

    # ── Exact SHAP ────────────────────────────────────────────────────────────

    def exact_shap_row(
        self,
        x0: pd.Series | pd.DataFrame,
        features: Optional[list[str]] = None,
        top_features: Optional[int] = 6,
    ) -> pd.Series:
        """Exact Shapley values via combinatorial definition. Use only for small feature subsets."""
        x0 = self._ensure_series(x0)
        features = self._resolve_feature_subset(features=features, top_features=top_features)
        M = len(features)
        factorial_M = math.factorial(M)
        phi = {f: 0.0 for f in features}
        cache: dict[frozenset, float] = {frozenset(): self._base_value_cached}

        def v(S: set) -> float:
            key = frozenset(S)
            if key not in cache:
                cache[key] = self._coalition_value(x0, set(key))
            return cache[key]

        for j in features:
            others = [f for f in features if f != j]
            for r in range(len(others) + 1):
                for subset in itertools.combinations(others, r):
                    S = set(subset)
                    weight = math.factorial(len(S)) * math.factorial(M - len(S) - 1) / factorial_M
                    phi[j] += weight * (v(S | {j}) - v(S))

        return pd.Series(phi).sort_values(key=np.abs, ascending=False)

    # ── Permutation SHAP ─────────────────────────────────────────────────────

    def permutation_shap_row(
        self,
        x0: pd.Series | pd.DataFrame,
        features: Optional[list[str]] = None,
        top_features: Optional[int] = 15,
        n_permutations: int = 200,
        random_state: Optional[int] = None,
    ) -> pd.Series:
        """Approximate SHAP via random feature permutations."""
        x0 = self._ensure_series(x0)
        features = self._resolve_feature_subset(features=features, top_features=top_features)
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        phi = {f: 0.0 for f in features}

        for _ in range(n_permutations):
            perm = list(rng.permutation(features))
            X_current = self.background.copy()
            prev_value = self._base_value_cached
            for f in perm:
                X_current[f] = x0[f]
                new_value = float(self._predict_df(X_current).mean())
                phi[f] += new_value - prev_value
                prev_value = new_value

        for f in phi:
            phi[f] /= n_permutations

        return pd.Series(phi).sort_values(key=np.abs, ascending=False)

    # ── Batch explain ─────────────────────────────────────────────────────────

    def explain_many(
        self,
        X: pd.DataFrame,
        method: str = "permutation",
        features: Optional[list[str]] = None,
        top_features: Optional[int] = 15,
        n_permutations: int = 100,
        exact_top_features: int = 6,
    ) -> pd.DataFrame:
        """Explain multiple rows; returns DataFrame of SHAP values (rows=instances, cols=features)."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if method == "permutation":
            resolved = self._resolve_feature_subset(features=features, top_features=top_features)
        elif method == "exact":
            resolved = self._resolve_feature_subset(features=features, top_features=exact_top_features)
        else:
            raise ValueError("method must be 'permutation' or 'exact'")

        results = []
        for i in range(len(X)):
            row = X.iloc[i]
            if method == "permutation":
                phi = self.permutation_shap_row(
                    x0=row, features=resolved, top_features=None,
                    n_permutations=n_permutations, random_state=self.random_state + i,
                )
            else:
                phi = self.exact_shap_row(x0=row, features=resolved, top_features=None)
            results.append(phi)

        shap_df = pd.DataFrame(results).fillna(0.0)
        shap_df.index = X.index
        return shap_df

    # ── Base value and additivity ─────────────────────────────────────────────

    def base_value(self) -> float:
        """Expected prediction under the background distribution (v(∅))."""
        return self._base_value_cached

    def reconstruct_prediction(self, shap_values_row: pd.Series) -> float:
        return self._base_value_cached + float(shap_values_row.sum())

    # ── Global importance ─────────────────────────────────────────────────────

    def summary_importance(self, shap_df: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.DataFrame({
                "feature": shap_df.columns,
                "mean_abs_shap": np.abs(shap_df).mean(axis=0).values,
                "mean_shap": shap_df.mean(axis=0).values,
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    # ── Plots (matplotlib, used by notebooks) ────────────────────────────────

    def plot_summary(
        self,
        shap_df: pd.DataFrame,
        X_values: pd.DataFrame,
        top_n: int = 20,
        title: str = "Manual SHAP summary",
        save_path: Optional[str] = None,
        figsize: tuple = (10, 7),
    ) -> None:
        common_cols = [c for c in shap_df.columns if c in X_values.columns]
        if not common_cols:
            raise ValueError("No overlapping columns between shap_df and X_values")
        shap_df = shap_df[common_cols].copy()
        X_values = X_values[common_cols].copy()

        imp = np.abs(shap_df).mean(axis=0).sort_values(ascending=False)
        selected = list(imp.head(top_n).index)
        order_reversed = selected[::-1]

        plt.figure(figsize=figsize)
        rng = np.random.default_rng(self.random_state)
        for yi, feat in enumerate(order_reversed):
            shap_col = shap_df[feat].to_numpy()
            feat_vals = X_values[feat].to_numpy()
            feat_range = np.nanmax(feat_vals) - np.nanmin(feat_vals)
            color_vals = (
                (feat_vals - np.nanmin(feat_vals)) / (feat_range + 1e-12)
                if feat_range > 1e-12
                else np.full(len(feat_vals), 0.5)
            )
            jitter = rng.normal(0, 0.10, size=len(shap_col))
            plt.scatter(
                shap_col, np.full(len(shap_col), yi) + jitter,
                c=color_vals, cmap="coolwarm", alpha=0.7, s=18, edgecolors="none",
            )
        plt.axvline(0.0, linewidth=1)
        plt.yticks(range(len(order_reversed)), order_reversed)
        plt.xlabel("Contribution to prediction")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.show()

    def plot_local(
        self,
        shap_row: pd.Series,
        x0: Optional[pd.Series] = None,
        top_n: int = 10,
        title: str = "Manual SHAP local explanation",
        save_path: Optional[str] = None,
        figsize: tuple = (9, 5),
    ) -> None:
        plot_series = shap_row.sort_values(key=np.abs, ascending=False).head(top_n).sort_values()
        plt.figure(figsize=figsize)
        plt.barh(plot_series.index, plot_series.values)
        plt.axvline(0.0, linewidth=1)
        plt.xlabel("Contribution to prediction")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.show()

    def explain_row_report(
        self,
        x0: pd.Series | pd.DataFrame,
        method: str = "permutation",
        features: Optional[list[str]] = None,
        top_features: Optional[int] = 15,
        n_permutations: int = 200,
    ) -> pd.DataFrame:
        x0_series = self._ensure_series(x0)
        if method == "permutation":
            phi = self.permutation_shap_row(
                x0=x0_series, features=features, top_features=top_features,
                n_permutations=n_permutations,
            )
        elif method == "exact":
            phi = self.exact_shap_row(x0=x0_series, features=features, top_features=top_features)
        else:
            raise ValueError("method must be 'permutation' or 'exact'")

        report = pd.DataFrame({
            "feature": phi.index,
            "phi": phi.values,
            "x0_value": [x0_series[f] for f in phi.index],
        })
        report["abs_phi"] = report["phi"].abs()
        report["direction"] = np.where(report["phi"] >= 0, "increase", "decrease")
        return report.sort_values("abs_phi", ascending=False).reset_index(drop=True)
