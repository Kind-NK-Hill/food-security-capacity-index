#!/usr/bin/env python3
"""Run the reproducible core analysis for the cereal-capacity case study.

Inputs must already have passed the official-data gate.  The script performs
country-profile clustering, one OLS association model, one shallow random
forest, cross-fitted explanation diagnostics, and descriptive outcome
alignment.  It deliberately contains no t-SNE, model-family sweep, tuning,
forecasting, Category ICE/PDP, or SHAP dependence gallery.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.cluster.hierarchy import cut_tree, dendrogram, leaves_list, linkage
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "capacity_v2_rank_50_50"
BASELINE_VALUE = 0.50
N_FOLDS = 5
N_CLUSTER_BOOTSTRAPS = 500
RANDOM_STATE = 42
FEATURES = [
    "electricity_access_pct",
    "rural_population_pct",
    "rural_population_growth_pct",
    "female_agricultural_employment_pct",
    "arable_land_hectares_per_person",
    "fertilizer_consumption_kg_per_hectare_arable_land",
]
FEATURE_LABELS = {
    "electricity_access_pct": "Electricity access",
    "rural_population_pct": "Rural population share",
    "rural_population_growth_pct": "Rural population growth",
    "female_agricultural_employment_pct": "Female agricultural employment",
    "arable_land_hectares_per_person": "Arable land per person",
    "fertilizer_consumption_kg_per_hectare_arable_land": "Fertilizer intensity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the country-grouped cereal-capacity analysis."
    )
    parser.add_argument(
        "--run-date",
        default=date.today().isoformat(),
        help="Output folder label in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--explanatory-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "explanatory",
        help="Directory containing explanatory_panel.csv, summary.json, manifest.json, and feature_dictionary.csv.",
    )
    parser.add_argument(
        "--capacity-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "capacity",
        help="Directory containing capacity_v2_panel.csv and its sensitivity summaries.",
    )
    parser.add_argument(
        "--outcomes-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "outcomes" / "outcome_panel.csv",
        help="Processed PoU/FIES panel joined by ISO3 and year. GHI columns are optional.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory. Defaults to results/generated/<run-date>.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to figures/generated/<run-date>.",
    )
    parser.add_argument(
        "--cluster-bootstraps",
        type=int,
        default=N_CLUSTER_BOOTSTRAPS,
        help="Within-country year bootstrap replicates (default: 500).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.15g",
    )
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def portable_path(path: Path) -> str:
    """Return a project-relative path when possible, avoiding machine-specific manifests."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def input_record(path: Path) -> dict[str, Any]:
    return {
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def correlation(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.unique(left[finite]).size < 2 or np.unique(right[finite]).size < 2:
        return float("nan")
    result = spearmanr(left, right, nan_policy="omit")
    return float(result.statistic)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": correlation(y_true, y_pred),
    }


def retained_features(summary_path: Path) -> list[str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary["data_gate"]["modeling_allowed"]:
        raise ValueError("Official explanatory-data gate does not allow modeling")
    retained = list(summary["data_gate"]["passed_features"])
    unexpected = sorted(set(retained) - set(FEATURES))
    if unexpected:
        raise ValueError(f"Data gate returned unexpected features: {unexpected}")
    return [feature for feature in FEATURES if feature in retained]


def validate_explanatory_gate_and_manifest(
    panel: pd.DataFrame,
    panel_path: Path,
    summary_path: Path,
    manifest_path: Path,
) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_outputs = {
        Path(record["path"]).name: record for record in manifest.get("outputs", [])
    }
    for path in [panel_path, summary_path]:
        record = declared_outputs.get(path.name)
        if record is None:
            raise ValueError(f"Explanatory manifest does not declare {path.name}")
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
            raise ValueError(
                f"Explanatory artifact hash mismatch for {path.name}: "
                f"manifest={record.get('sha256')}, actual={actual_hash}"
            )

    summary_features = retained_features(summary_path)
    recomputed_features: list[str] = []
    for feature in FEATURES:
        if feature not in panel.columns:
            continue
        overall_coverage = float(panel[feature].notna().mean())
        minimum_country_years = int(panel.groupby("iso3")[feature].count().min())
        if overall_coverage >= 0.80 and minimum_country_years >= 8:
            recomputed_features.append(feature)
    if summary_features != recomputed_features:
        raise ValueError(
            "Explanatory summary and actual panel disagree on the data gate: "
            f"summary={summary_features}, recomputed={recomputed_features}"
        )
    if recomputed_features != FEATURES:
        raise ValueError(
            "The locked six-feature core analysis cannot run with a reduced feature set. "
            f"Passed={recomputed_features}; a new user adjudication is required."
        )
    return recomputed_features


def build_country_index_summary(
    capacity: pd.DataFrame, stability: pd.DataFrame
) -> pd.DataFrame:
    grouped = (
        capacity.groupby(["iso3", "country", "legacy_category"], as_index=False)
        .agg(
            median_capacity_v2=(TARGET, "median"),
            minimum_annual_capacity_v2=(TARGET, "min"),
            maximum_annual_capacity_v2=(TARGET, "max"),
            median_ssr_percentile_rank=("ssr_percentile_rank", "median"),
            median_production_per_person_percentile_rank=(
                "production_per_person_percentile_rank",
                "median",
            ),
            annual_score_sd=(TARGET, lambda values: float(np.std(values, ddof=0))),
        )
        .sort_values(["median_capacity_v2", "iso3"], ascending=[False, True])
        .reset_index(drop=True)
    )
    grouped["country_rank"] = grouped["median_capacity_v2"].rank(
        method="average", ascending=False
    )
    stability_columns = [
        "iso3",
        "central_weight_rank_min",
        "central_weight_rank_max",
        "central_weight_rank_range",
        "all_weight_rank_min",
        "all_weight_rank_max",
        "all_weight_rank_range",
    ]
    result = grouped.merge(
        stability[stability_columns], on="iso3", how="left", validate="one_to_one"
    )
    if result[stability_columns[1:]].isna().any().any():
        raise ValueError("Capacity rank-stability summary did not join all 30 countries")
    return result.sort_values(["country_rank", "iso3"], ignore_index=True)


def index_robustness(
    capacity: pd.DataFrame, weight_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = capacity.groupby("iso3").agg(
        primary=(TARGET, "median"),
        legacy_weights=("capacity_v2_rank_legacy_56_44", "median"),
        minmax=("capacity_v2_minmax_50_50", "median"),
    )
    central = weight_summary.loc[weight_summary["ssr_weight"].between(0.40, 0.60)]
    selected = weight_summary.loc[np.isclose(weight_summary["ssr_weight"], 0.50)]
    if len(selected) != 1 or len(central) != 21:
        raise ValueError("Capacity v2 weight-sweep grid is incomplete")
    row = selected.iloc[0]
    robustness = pd.DataFrame(
        [
            {
                "diagnostic": "country-median rank: 50/50 versus 56/44",
                "value": correlation(medians["primary"], medians["legacy_weights"]),
                "interpretation": "Spearman rank agreement",
            },
            {
                "diagnostic": "country-median rank: percentile-rank versus annual min-max",
                "value": correlation(medians["primary"], medians["minmax"]),
                "interpretation": "Spearman rank agreement",
            },
            {
                "diagnostic": "minimum rank agreement over central SSR weights 0.40-0.60",
                "value": float(central["spearman_country_rank_vs_50_50"].min()),
                "interpretation": "minimum Spearman versus 50/50",
            },
            {
                "diagnostic": "minimum rank agreement over all 101 SSR weights",
                "value": float(weight_summary["spearman_country_rank_vs_50_50"].min()),
                "interpretation": "minimum Spearman versus 50/50",
            },
            {
                "diagnostic": "weight combinations satisfying all four category hypotheses",
                "value": int(weight_summary["all_category_constraints_pass"].sum()),
                "interpretation": "count out of 101; weights are not selected to repair failure",
            },
        ]
    )
    constraint_map = [
        (
            "Category 1 above Category 2",
            "margin_min_category1_minus_max_category2",
        ),
        (
            "Category 2 above Category 3",
            "margin_min_category2_minus_max_category3",
        ),
        (
            "Flexible 1 above Category 3",
            "margin_min_flexible1_minus_max_category3",
        ),
        (
            "Flexible 2 below Category 1",
            "margin_min_category1_minus_max_flexible2",
        ),
    ]
    constraints = pd.DataFrame(
        [
            {
                "expert_hypothesis": label,
                "margin_at_50_50": float(row[column]),
                "passes_strict_ordering": bool(row[column] > 0),
                "status": "supported" if row[column] > 0 else "contradicted",
            }
            for label, column in constraint_map
        ]
    )
    return robustness, constraints


def country_structure(
    panel: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, bool]:
    profiles = (
        panel.groupby(["iso3", "country", "legacy_category"], as_index=False)[features]
        .median()
        .sort_values("iso3", ignore_index=True)
    )
    if profiles[features].isna().any().any():
        raise ValueError("A retained feature has an undefined country median")
    scaler = RobustScaler()
    scaled = scaler.fit_transform(profiles[features])
    tree = linkage(scaled, method="ward", metric="euclidean", optimal_ordering=True)

    silhouette_records: list[dict[str, Any]] = []
    for k in range(2, 7):
        labels = cut_tree(tree, n_clusters=[k]).reshape(-1) + 1
        silhouette_records.append(
            {
                "k": k,
                "silhouette_score": float(silhouette_score(scaled, labels, metric="euclidean")),
            }
        )
    silhouette = pd.DataFrame(silhouette_records)
    selected_k = int(
        silhouette.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]["k"]
    )
    full_labels = cut_tree(tree, n_clusters=[selected_k]).reshape(-1) + 1
    if len(np.unique(full_labels)) != selected_k:
        raise ValueError("Ward tree did not yield the selected number of clusters")

    rng = np.random.default_rng(RANDOM_STATE)
    stability_records: list[dict[str, Any]] = []
    country_groups = {
        iso3: group.sort_values("year", ignore_index=True)
        for iso3, group in panel.groupby("iso3", sort=True)
    }
    for bootstrap in range(1, N_CLUSTER_BOOTSTRAPS + 1):
        boot_profiles: list[np.ndarray] = []
        redraw_count = 0
        for iso3 in profiles["iso3"]:
            group = country_groups[iso3]
            for attempt in range(1, 101):
                sampled = rng.integers(0, len(group), size=len(group))
                values = group.iloc[sampled][features].to_numpy(dtype=float)
                with np.errstate(all="ignore"):
                    medians = np.nanmedian(values, axis=0)
                if np.isfinite(medians).all():
                    redraw_count += attempt - 1
                    break
            else:
                raise ValueError(
                    f"Could not obtain finite bootstrap medians for {iso3} after 100 draws"
                )
            boot_profiles.append(medians)
        boot_scaled = RobustScaler().fit_transform(np.vstack(boot_profiles))
        boot_tree = linkage(
            boot_scaled, method="ward", metric="euclidean", optimal_ordering=True
        )
        boot_labels = cut_tree(boot_tree, n_clusters=[selected_k]).reshape(-1) + 1
        stability_records.append(
            {
                "bootstrap_replicate": bootstrap,
                "selected_k_fixed": selected_k,
                "country_draw_redraws_for_finite_medians": redraw_count,
                "adjusted_rand_index_vs_full": float(
                    adjusted_rand_score(full_labels, boot_labels)
                ),
            }
        )
    stability = pd.DataFrame(stability_records)
    median_ari = float(stability["adjusted_rand_index_vs_full"].median())
    stability_gate = median_ari >= 0.60

    membership = profiles[["iso3", "country", "legacy_category"]].copy()
    membership["selected_k"] = selected_k
    membership["ward_cluster"] = full_labels
    membership["median_bootstrap_ari"] = median_ari
    membership["main_text_stability_gate_pass"] = stability_gate
    for column_index, feature in enumerate(features):
        membership[f"robust_scaled__{feature}"] = scaled[:, column_index]
    return membership, silhouette, stability, tree, scaled, stability_gate


def fit_fold_preprocessing(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    features: list[str],
    missing_indicator_features: list[str],
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], SimpleImputer, StandardScaler | None]:
    """Fit medians in the training fold and append globally fixed indicators."""
    imputer = SimpleImputer(
        strategy="median", add_indicator=False, keep_empty_features=True
    )
    train = imputer.fit_transform(x_train[features])
    test = imputer.transform(x_test[features])
    names = list(features)
    if missing_indicator_features:
        train_indicators = x_train[missing_indicator_features].isna().to_numpy(dtype=float)
        test_indicators = x_test[missing_indicator_features].isna().to_numpy(dtype=float)
        train = np.column_stack([train, train_indicators])
        test = np.column_stack([test, test_indicators])
        names.extend(
            [f"missingindicator_{feature}" for feature in missing_indicator_features]
        )
    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    return train, test, names, imputer, scaler


def grouped_permutation_mae_importance(
    model: RandomForestRegressor,
    x_test: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    transformed_feature_names: list[str],
    missing_indicator_features: list[str],
    random_state: int,
    repeats: int = 30,
) -> pd.DataFrame:
    """Jointly permute a value column and its indicator to preserve valid pairs."""
    baseline_mae = float(mean_absolute_error(y_test, model.predict(x_test)))
    rng = np.random.default_rng(random_state)
    records: list[dict[str, Any]] = []
    for feature in features:
        column_indices = [transformed_feature_names.index(feature)]
        indicator_name = f"missingindicator_{feature}"
        if feature in missing_indicator_features:
            column_indices.append(transformed_feature_names.index(indicator_name))
        deltas: list[float] = []
        for _ in range(repeats):
            permutation = rng.permutation(len(x_test))
            permuted = x_test.copy()
            permuted[:, column_indices] = x_test[permutation][:, column_indices]
            permuted_mae = float(mean_absolute_error(y_test, model.predict(permuted)))
            deltas.append(permuted_mae - baseline_mae)
        records.append(
            {
                "feature": feature,
                "permutation_group_columns": ";".join(
                    transformed_feature_names[index] for index in column_indices
                ),
                "grouped_permutation_importance_mean_mae_increase": float(
                    np.mean(deltas)
                ),
                "grouped_permutation_importance_sd": float(np.std(deltas, ddof=0)),
                "permutation_repeats": repeats,
            }
        )
    result = pd.DataFrame(records)
    result["importance_rank"] = result[
        "grouped_permutation_importance_mean_mae_increase"
    ].rank(method="average", ascending=False)
    result["top_five"] = result["importance_rank"].le(5)
    return result


def fold_assignments(panel: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    splitter = GroupKFold(n_splits=N_FOLDS)
    groups = panel["iso3"].to_numpy()
    splits = list(splitter.split(panel[[TARGET]], panel[TARGET], groups=groups))
    fold_ids = np.zeros(len(panel), dtype=int)
    seen_test_countries: set[str] = set()
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_countries = set(groups[train_index])
        test_countries = set(groups[test_index])
        if train_countries & test_countries:
            raise AssertionError(f"Country leakage in grouped fold {fold}")
        if seen_test_countries & test_countries:
            raise AssertionError("A country appears in more than one held-out fold")
        seen_test_countries |= test_countries
        fold_ids[test_index] = fold
    if len(seen_test_countries) != panel["iso3"].nunique() or (fold_ids == 0).any():
        raise AssertionError("Grouped folds do not cover every country-year row exactly once")
    return splits, fold_ids


def run_models(
    panel: pd.DataFrame, features: list[str]
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    try:
        import shap
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TreeSHAP dependency is missing. Run this script with the project-local "
            ".venv created from requirements-analysis.txt."
        ) from error

    panel = panel.sort_values(["iso3", "year"], ignore_index=True).copy()
    splits, fold_ids = fold_assignments(panel)
    x = panel[features]
    missing_indicator_features = [feature for feature in features if x[feature].isna().any()]
    y = panel[TARGET].to_numpy(dtype=float)
    groups = panel["iso3"].to_numpy()
    baseline_predictions = np.full(len(panel), BASELINE_VALUE, dtype=float)
    ols_predictions = np.full(len(panel), np.nan, dtype=float)
    rf_predictions = np.full(len(panel), np.nan, dtype=float)
    ols_fold_coefficients: list[dict[str, Any]] = []
    rf_fold_importance: list[dict[str, Any]] = []
    rf_missingness_records: list[dict[str, Any]] = []
    shap_records: list[dict[str, Any]] = []
    fold_metric_records: list[dict[str, Any]] = []
    fold_country_records: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        train_countries = sorted(set(groups[train_index]))
        test_countries = sorted(set(groups[test_index]))
        fold_country_records.append(
            {
                "fold": fold,
                "train_country_count": len(train_countries),
                "test_country_count": len(test_countries),
                "train_countries": ";".join(train_countries),
                "test_countries": ";".join(test_countries),
                "country_overlap_count": len(set(train_countries) & set(test_countries)),
            }
        )

        x_train_ols, x_test_ols, ols_names, _, _ = fit_fold_preprocessing(
            x_train,
            x_test,
            features,
            missing_indicator_features,
            standardize=True,
        )
        ols = LinearRegression().fit(x_train_ols, y_train)
        ols_predictions[test_index] = ols.predict(x_test_ols)
        coefficient_map = dict(zip(ols_names, ols.coef_, strict=True))
        for feature in features:
            ols_fold_coefficients.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "coefficient": float(coefficient_map[feature]),
                    "sign": (
                        "positive"
                        if coefficient_map[feature] > 0
                        else "negative" if coefficient_map[feature] < 0 else "zero"
                    ),
                }
            )

        x_train_rf, x_test_rf, rf_names, _, _ = fit_fold_preprocessing(
            x_train,
            x_test,
            features,
            missing_indicator_features,
            standardize=False,
        )
        rf = RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(x_train_rf, y_train)
        rf_predictions[test_index] = rf.predict(x_test_rf)

        fold_importance = grouped_permutation_mae_importance(
            rf,
            x_test_rf,
            y_test,
            features,
            rf_names,
            missing_indicator_features,
            random_state=RANDOM_STATE + fold,
            repeats=30,
        )

        explainer = shap.TreeExplainer(rf)
        shap_values_raw = explainer.shap_values(x_test_rf, check_additivity=False)
        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) != 1:
                raise ValueError("Unexpected multi-output SHAP result for regression")
            shap_values = np.asarray(shap_values_raw[0], dtype=float)
        elif hasattr(shap_values_raw, "values"):
            shap_values = np.asarray(shap_values_raw.values, dtype=float)
        else:
            shap_values = np.asarray(shap_values_raw, dtype=float)
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]
        if shap_values.shape != x_test_rf.shape:
            raise ValueError(
                f"Unexpected SHAP shape {shap_values.shape}; expected {x_test_rf.shape}"
            )

        for column, transformed_feature in enumerate(rf_names):
            direction = (
                "missingness_indicator"
                if transformed_feature.startswith("missingindicator_")
                else "base_feature"
            )
            for local_row, panel_row in enumerate(test_index):
                shap_records.append(
                    {
                        "row_id": int(panel_row),
                        "iso3": panel.iloc[panel_row]["iso3"],
                        "country": panel.iloc[panel_row]["country"],
                        "year": int(panel.iloc[panel_row]["year"]),
                        "fold": fold,
                        "feature": transformed_feature,
                        "column_role": direction,
                        "feature_value_after_fold_imputation": float(
                            x_test_rf[local_row, column]
                        ),
                        "shap_value": float(shap_values[local_row, column]),
                    }
                )
        direction_by_feature: dict[str, tuple[float, str]] = {}
        for feature in features:
            column = rf_names.index(feature)
            observed = x_test[feature].notna().to_numpy()
            if observed.sum() >= 3 and np.unique(x_test_rf[observed, column]).size >= 2:
                rho = correlation(
                    x_test_rf[observed, column], shap_values[observed, column]
                )
            else:
                rho = math.nan
            direction = (
                "positive"
                if np.isfinite(rho) and rho >= 0.10
                else "negative"
                if np.isfinite(rho) and rho <= -0.10
                else "neutral_or_weak"
            )
            direction_by_feature[feature] = (rho, direction)
        total_abs_shap = float(np.abs(shap_values).sum())
        for feature in missing_indicator_features:
            indicator_name = f"missingindicator_{feature}"
            indicator_column = rf_names.index(indicator_name)
            indicator_abs_shap = float(np.abs(shap_values[:, indicator_column]).sum())
            rf_missingness_records.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "indicator_name": indicator_name,
                    "heldout_missing_rows": int(x_test[feature].isna().sum()),
                    "heldout_rows": len(x_test),
                    "mean_absolute_indicator_shap": float(
                        np.abs(shap_values[:, indicator_column]).mean()
                    ),
                    "indicator_share_of_all_absolute_shap": (
                        indicator_abs_shap / total_abs_shap if total_abs_shap > 0 else 0.0
                    ),
                    "audit_note": "reported separately; value and indicator were jointly permuted for conceptual-feature importance",
                }
            )
        for record in fold_importance.to_dict(orient="records"):
            rho, direction = direction_by_feature[record["feature"]]
            rf_fold_importance.append(
                {
                    "fold": fold,
                    **record,
                    "heldout_feature_shap_spearman": rho,
                    "heldout_shap_direction": direction,
                    "missingness_indicator_in_model": record["feature"]
                    in missing_indicator_features,
                }
            )

        for model_name, prediction in [
            ("constant_0.50_baseline", baseline_predictions[test_index]),
            ("ols", ols_predictions[test_index]),
            ("shallow_random_forest", rf_predictions[test_index]),
        ]:
            fold_metric_records.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "heldout_rows": len(test_index),
                    "heldout_countries": len(test_countries),
                    **metrics(y_test, prediction),
                }
            )

    if np.isnan(ols_predictions).any() or np.isnan(rf_predictions).any():
        raise AssertionError("OOF prediction vectors are incomplete")

    oof = panel[["iso3", "country", "year", "legacy_category", TARGET]].copy()
    oof["fold"] = fold_ids
    oof["constant_0_50_prediction"] = baseline_predictions
    oof["ols_prediction"] = ols_predictions
    oof["shallow_random_forest_prediction"] = rf_predictions
    oof["ols_residual"] = y - ols_predictions
    oof["shallow_random_forest_residual"] = y - rf_predictions

    metric_records: list[dict[str, Any]] = []
    for model_name, predictions in [
        ("constant_0.50_baseline", baseline_predictions),
        ("ols", ols_predictions),
        ("shallow_random_forest", rf_predictions),
    ]:
        metric_records.append(
            {
                "model": model_name,
                "evaluation": "five-fold country-grouped out-of-fold",
                "rows": len(panel),
                "countries": int(panel["iso3"].nunique()),
                **metrics(y, predictions),
            }
        )
    model_metrics = pd.DataFrame(metric_records)
    fold_metrics = pd.DataFrame(fold_metric_records)
    fold_countries = pd.DataFrame(fold_country_records)

    full_standardized, _, full_names, _, _ = fit_fold_preprocessing(
        x,
        x,
        features,
        missing_indicator_features,
        standardize=True,
    )
    design = sm.add_constant(full_standardized, has_constant="add")
    design_names = ["intercept", *full_names]
    ols_full = sm.OLS(y, design).fit(
        cov_type="cluster", cov_kwds={"groups": panel["iso3"].to_numpy()}
    )
    confidence = np.asarray(ols_full.conf_int(alpha=0.05))
    linear_coefficients = pd.DataFrame(
        {
            "term": design_names,
            "coefficient": np.asarray(ols_full.params),
            "country_clustered_standard_error": np.asarray(ols_full.bse),
            "ci_95_lower": confidence[:, 0],
            "ci_95_upper": confidence[:, 1],
            "p_value_descriptive_only": np.asarray(ols_full.pvalues),
            "is_missingness_indicator": [
                False, *[name.startswith("missingindicator_") for name in full_names]
            ],
            "interpretation_unit": [
                "target intercept",
                *[
                    "Capacity v2 units per one full-panel predictor standard deviation"
                    for _ in full_names
                ],
            ],
        }
    )

    fold_coefficients = pd.DataFrame(ols_fold_coefficients)
    coefficient_stability_records: list[dict[str, Any]] = []
    for feature, group in fold_coefficients.groupby("feature", sort=False):
        positive = int(group["coefficient"].gt(0).sum())
        negative = int(group["coefficient"].lt(0).sum())
        zero = int(group["coefficient"].eq(0).sum())
        majority = "positive" if positive > negative else "negative" if negative > positive else "tie"
        coefficient_stability_records.append(
            {
                "feature": feature,
                "folds": len(group),
                "positive_folds": positive,
                "negative_folds": negative,
                "zero_folds": zero,
                "majority_sign": majority,
                "majority_sign_share": max(positive, negative, zero) / len(group),
                "mean_fold_coefficient": float(group["coefficient"].mean()),
                "minimum_fold_coefficient": float(group["coefficient"].min()),
                "maximum_fold_coefficient": float(group["coefficient"].max()),
            }
        )
    coefficient_stability = pd.DataFrame(coefficient_stability_records)

    rf_fold = pd.DataFrame(rf_fold_importance)
    base_rf_fold = rf_fold.copy()
    missingness_columns = [
        "fold",
        "feature",
        "indicator_name",
        "heldout_missing_rows",
        "heldout_rows",
        "mean_absolute_indicator_shap",
        "indicator_share_of_all_absolute_shap",
        "audit_note",
    ]
    missingness_fold = pd.DataFrame(rf_missingness_records, columns=missingness_columns)
    explanation_records: list[dict[str, Any]] = []
    for feature, group in base_rf_fold.groupby("feature", sort=False):
        positive = int(group["heldout_shap_direction"].eq("positive").sum())
        negative = int(group["heldout_shap_direction"].eq("negative").sum())
        neutral = int(group["heldout_shap_direction"].eq("neutral_or_weak").sum())
        contradictory = positive > 0 and negative > 0
        top_five_count = int(group["top_five"].fillna(False).sum())
        stable = len(features) == 6 and top_five_count >= 4 and not contradictory
        missing_group = missingness_fold.loc[missingness_fold["feature"].eq(feature)]
        explanation_records.append(
            {
                "feature": feature,
                "folds": len(group),
                "mean_heldout_grouped_permutation_importance_mae_increase": float(
                    group["grouped_permutation_importance_mean_mae_increase"].mean()
                ),
                "sd_across_fold_importance_means": float(
                    group["grouped_permutation_importance_mean_mae_increase"].std(ddof=0)
                ),
                "median_importance_rank": float(group["importance_rank"].median()),
                "top_five_fold_count": top_five_count,
                "positive_direction_folds": positive,
                "negative_direction_folds": negative,
                "neutral_or_weak_direction_folds": neutral,
                "median_heldout_feature_shap_spearman": float(
                    group["heldout_feature_shap_spearman"].median()
                ),
                "missingness_indicator_present": feature in missing_indicator_features,
                "mean_indicator_share_of_all_absolute_shap": (
                    float(missing_group["indicator_share_of_all_absolute_shap"].mean())
                    if len(missing_group)
                    else 0.0
                ),
                "maximum_indicator_share_of_all_absolute_shap": (
                    float(missing_group["indicator_share_of_all_absolute_shap"].max())
                    if len(missing_group)
                    else 0.0
                ),
                "direction_contradictory_across_folds": contradictory,
                "stable_for_main_text": stable,
                "stability_rule": "exactly six prespecified features; top five in >=4/5 folds; no positive/negative observed-value SHAP direction contradiction at |rho|>=0.10",
            }
        )
    explanation_stability = pd.DataFrame(explanation_records).sort_values(
        ["mean_heldout_grouped_permutation_importance_mae_increase", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )

    rank_pivot = base_rf_fold.pivot(index="feature", columns="fold", values="importance_rank")
    top_pivot = base_rf_fold.pivot(index="feature", columns="fold", values="top_five")
    pair_records: list[dict[str, Any]] = []
    for first, second in itertools.combinations(range(1, N_FOLDS + 1), 2):
        first_top = set(top_pivot.index[top_pivot[first].fillna(False)])
        second_top = set(top_pivot.index[top_pivot[second].fillna(False)])
        union = first_top | second_top
        pair_records.append(
            {
                "fold_1": first,
                "fold_2": second,
                "spearman_feature_rank": correlation(rank_pivot[first], rank_pivot[second]),
                "top_five_overlap_count": len(first_top & second_top),
                "top_five_jaccard": len(first_top & second_top) / len(union),
            }
        )
    rank_pairwise = pd.DataFrame(pair_records)
    shap_values = pd.DataFrame(shap_records)
    return (
        oof,
        model_metrics,
        fold_metrics,
        fold_countries,
        linear_coefficients,
        coefficient_stability,
        rf_fold,
        missingness_fold,
        explanation_stability,
        rank_pairwise,
        shap_values,
    )


def external_alignment(
    capacity: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = capacity[["iso3", "country", "year", TARGET]].rename(
        columns={"country": "country_capacity"}
    ).merge(
        outcomes.rename(columns={"country": "country_outcome"}),
        on=["iso3", "year"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(capacity):
        raise ValueError(
            f"Outcome source did not cover every Capacity v2 country-year key: "
            f"capacity={len(capacity)}, merged={len(merged)}"
        )
    mismatched_names = merged.loc[
        merged["country_capacity"].ne(merged["country_outcome"]),
        ["iso3", "country_capacity", "country_outcome"],
    ].drop_duplicates()
    if len(mismatched_names):
        raise ValueError(f"Country display-name mismatch after ISO3-year join:\n{mismatched_names}")
    merged["country"] = merged["country_capacity"]
    views = [
        {
            "outcome_view": "pou_lower_bound",
            "column": "pou_lower_bound",
            "status": "pou_value_status",
            "period": "pou_year_period",
            "valid": {"observed_exact", "left_censored"},
            "expected_direction": "negative",
            "report_role": "main",
            "semantics": "official PoU three-year-average lower-bound view; '<2.5' contributes 0 only as a bound",
        },
        {
            "outcome_view": "pou_upper_bound",
            "column": "pou_upper_bound",
            "status": "pou_value_status",
            "period": "pou_year_period",
            "valid": {"observed_exact", "left_censored"},
            "expected_direction": "negative",
            "report_role": "main",
            "semantics": "official PoU three-year-average upper-bound view; '<2.5' contributes 2.5 only as a bound",
        },
        {
            "outcome_view": "fies_exact",
            "column": "fies_exact_value",
            "status": "fies_value_status",
            "period": "fies_year_period",
            "valid": {"observed_exact"},
            "expected_direction": "negative",
            "report_role": "main",
            "semantics": "published exact moderate-or-severe FIES three-year averages only",
        },
        {
            "outcome_view": "ghi_2025_edition_lower_bound",
            "column": "ghi_lower_bound",
            "status": "ghi_value_status",
            "period": None,
            "valid": {"observed_exact", "left_censored"},
            "expected_direction": "negative",
            "report_role": "appendix_sparse_sensitivity",
            "semantics": "same-edition sparse GHI reference years; no interpolation; lower-bound view",
        },
        {
            "outcome_view": "ghi_2025_edition_upper_bound",
            "column": "ghi_upper_bound",
            "status": "ghi_value_status",
            "period": None,
            "valid": {"observed_exact", "left_censored"},
            "expected_direction": "negative",
            "report_role": "appendix_sparse_sensitivity",
            "semantics": "same-edition sparse GHI reference years; no interpolation; upper-bound view",
        },
    ]
    required_main_columns = {
        "pou_lower_bound",
        "pou_upper_bound",
        "pou_value_status",
        "pou_year_period",
        "fies_exact_value",
        "fies_value_status",
        "fies_year_period",
    }
    if missing := sorted(required_main_columns - set(merged.columns)):
        raise ValueError(f"Outcome panel lacks required PoU/FIES columns: {missing}")
    available_views: list[dict[str, Any]] = []
    for view in views:
        columns_present = (
            view["column"] in merged.columns
            and view["status"] in merged.columns
            and (view["period"] is None or view["period"] in merged.columns)
        )
        if not columns_present:
            continue
        has_valid_rows = bool(
            (
                merged[view["status"]].isin(view["valid"])
                & merged[view["column"]].notna()
            ).any()
        )
        if view["report_role"] == "main" or has_valid_rows:
            available_views.append(view)
    views = available_views
    summary_records: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for view in views:
        valid_mask = (
            merged[view["status"]].isin(view["valid"])
            & merged[view["column"]].notna()
            & merged[TARGET].notna()
        )
        valid = merged.loc[valid_mask].copy()
        country = (
            valid.groupby(["iso3", "country"], as_index=False)
            .agg(
                overlapping_period_rows=("year", "count"),
                first_overlap_year=("year", "min"),
                last_overlap_year=("year", "max"),
                capacity_v2_median=(TARGET, "median"),
                outcome_median=(view["column"], "median"),
            )
            .sort_values("iso3", ignore_index=True)
        )
        minimum_expected_countries = 20 if view["report_role"] == "main" else 5
        if len(country) < minimum_expected_countries:
            raise ValueError(
                f"Unexpectedly sparse {view['outcome_view']} alignment: "
                f"{len(country)} countries, minimum {minimum_expected_countries}"
            )
        rho = correlation(country["capacity_v2_median"], country["outcome_median"])
        censored_rows = int(valid[view["status"]].eq("left_censored").sum())
        summary_records.append(
            {
                "outcome_view": view["outcome_view"],
                "report_role": view["report_role"],
                "country_count": len(country),
                "overlapping_country_period_rows": len(valid),
                "candidate_capacity_rows": len(merged),
                "excluded_missing_or_invalid_status_rows": int((~valid_mask).sum()),
                "left_censored_rows": censored_rows,
                "median_period_rows_per_country": float(
                    country["overlapping_period_rows"].median()
                ),
                "minimum_period_rows_per_country": int(
                    country["overlapping_period_rows"].min()
                ),
                "maximum_period_rows_per_country": int(
                    country["overlapping_period_rows"].max()
                ),
                "country_median_spearman": rho,
                "expected_direction": view["expected_direction"],
                "direction_consistent": (
                    rho < 0 if view["expected_direction"] == "negative" else rho > 0
                ),
                "semantics": view["semantics"],
                "inference": "descriptive construct alignment only; no p-value, prediction, or causal claim",
            }
        )
        for record in country.to_dict(orient="records"):
            country_rows = valid.loc[valid["iso3"].eq(record["iso3"])]
            status_counts = country_rows[view["status"]].value_counts().sort_index()
            detail_records.append(
                {
                    "outcome_view": view["outcome_view"],
                    **record,
                    "left_censored_period_rows": int(
                        country_rows[view["status"]].eq("left_censored").sum()
                    ),
                    "status_counts": ";".join(
                        f"{status}={count}" for status, count in status_counts.items()
                    ),
                }
            )
        for row in valid.to_dict(orient="records"):
            period = (
                row.get(view["period"], "")
                if view["period"] is not None
                else str(row["year"])
            )
            source_records.append(
                {
                    "outcome_view": view["outcome_view"],
                    "report_role": view["report_role"],
                    "iso3": row["iso3"],
                    "country": row["country"],
                    "year": int(row["year"]),
                    "source_period": period,
                    "capacity_v2_rank_50_50": row[TARGET],
                    "outcome_value_for_view": row[view["column"]],
                    "outcome_status": row[view["status"]],
                    "view_semantics": view["semantics"],
                }
            )
    return (
        pd.DataFrame(summary_records),
        pd.DataFrame(detail_records),
        pd.DataFrame(source_records),
    )


def plot_method_flow(path: Path) -> None:
    labels = [
        "Official data\n(WDI + FAOSTAT)",
        "Capacity v2\n(two transparent components)",
        "Expert hypotheses\n+ country structure",
        "Grouped OLS +\none shallow RF",
        "PoU/FIES\nconstruct alignment",
    ]
    fig, ax = plt.subplots(figsize=(12, 2.7), constrained_layout=True)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(-0.7, 0.7)
    ax.axis("off")
    colors = ["#E8F1F8", "#D8EAD3", "#FFF1CC", "#E8DDF1", "#F7D9D5"]
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        ax.text(
            index,
            0,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#555555"},
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(index + 0.68, 0),
                xytext=(index + 0.32, 0),
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.4},
            )
    ax.text(
        2,
        -0.56,
        "Association and robustness study — not a forecast and not causal",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_profile(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("country_rank", ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 10.2), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]}
    )
    axes[0].hlines(
        y,
        ordered["minimum_annual_capacity_v2"],
        ordered["maximum_annual_capacity_v2"],
        color="#A8A8A8",
        lw=2.1,
    )
    axes[0].scatter(
        ordered["median_capacity_v2"], y, color="#2166AC", s=28, zorder=3
    )
    axes[0].set(
        xlabel="Capacity v2 score (annual range and country median)",
        yticks=y,
        yticklabels=ordered["country"],
        xlim=(-0.02, 1.02),
    )
    axes[1].hlines(
        y,
        ordered["central_weight_rank_min"],
        ordered["central_weight_rank_max"],
        color="#A8A8A8",
        lw=2.1,
    )
    axes[1].scatter(
        ordered["country_rank"],
        y,
        label="Primary 50/50 rank",
        color="#C51B7D",
        s=27,
        zorder=3,
    )
    axes[1].set(
        xlabel="Country rank across central SSR weights (0.40–0.60)",
        xlim=(30.5, 0.5),
    )
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Capacity v2 country profiles, 2010-2023", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_country_structure(
    membership: pd.DataFrame,
    tree: np.ndarray,
    scaled: np.ndarray,
    features: list[str],
    path: Path,
    stable: bool,
) -> None:
    order = leaves_list(tree)
    ordered = membership.iloc[order].reset_index(drop=True)
    scaled_ordered = scaled[order, :]
    fig = plt.figure(figsize=(12.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.4, 5.2])
    ax_tree = fig.add_subplot(grid[0])
    dendrogram(
        tree,
        labels=membership["iso3"].tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=None,
        above_threshold_color="#555555",
        ax=ax_tree,
    )
    ax_tree.set_ylabel("Ward distance")
    ax_tree.spines[["top", "right"]].set_visible(False)
    ax_heat = fig.add_subplot(grid[1])
    sns.heatmap(
        scaled_ordered,
        cmap="vlag",
        center=0,
        yticklabels=[
            f"{country}  [C{cluster}]"
            for country, cluster in zip(
                ordered["country"], ordered["ward_cluster"], strict=True
            )
        ],
        xticklabels=[FEATURE_LABELS[feature] for feature in features],
        cbar_kws={"label": "Robust-scaled country median"},
        ax=ax_heat,
    )
    ax_heat.tick_params(axis="x", labelrotation=25)
    ax_heat.tick_params(axis="y", labelsize=8)
    title_status = "passes" if stable else "fails"
    fig.suptitle(
        f"Country profiles and exploratory Ward structure ({title_status} bootstrap stability gate)",
        fontsize=13,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_linear_coefficients(coefficients: pd.DataFrame, path: Path) -> None:
    data = coefficients.loc[coefficients["term"].isin(FEATURES)].copy()
    data["label"] = data["term"].map(FEATURE_LABELS)
    data = data.sort_values("coefficient")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    lower_error = data["coefficient"] - data["ci_95_lower"]
    upper_error = data["ci_95_upper"] - data["coefficient"]
    ax.errorbar(
        data["coefficient"],
        y,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        color="#2166AC",
        ecolor="#777777",
        capsize=3,
    )
    ax.axvline(0, color="#444444", lw=1, ls="--")
    ax.set(
        yticks=y,
        yticklabels=data["label"],
        xlabel="Capacity v2 association per predictor SD (95% country-clustered interval)",
        title="Standardized OLS associations",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shap_beeswarm(shap_values: pd.DataFrame, path: Path) -> None:
    data = shap_values.loc[shap_values["feature"].isin(FEATURES)].copy()
    importance_order = (
        data.assign(abs_shap=data["shap_value"].abs())
        .groupby("feature")["abs_shap"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    rng = np.random.default_rng(RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    color_values: list[float] = []
    scatter_handles = []
    for y_index, feature in enumerate(importance_order):
        feature_data = data.loc[data["feature"].eq(feature)].copy()
        raw = feature_data["feature_value_after_fold_imputation"].to_numpy(dtype=float)
        low, high = np.nanpercentile(raw, [5, 95])
        if high > low:
            normalized = np.clip((raw - low) / (high - low), 0, 1)
        else:
            normalized = np.full_like(raw, 0.5)
        jitter = rng.normal(0, 0.085, size=len(feature_data))
        handle = ax.scatter(
            feature_data["shap_value"],
            y_index + jitter,
            c=normalized,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=16,
            alpha=0.68,
            linewidths=0,
        )
        scatter_handles.append(handle)
        color_values.extend(normalized.tolist())
    ax.axvline(0, color="#555555", lw=0.9)
    ax.set(
        yticks=np.arange(len(importance_order)),
        yticklabels=[FEATURE_LABELS[feature] for feature in importance_order],
        xlabel="Cross-fitted SHAP value for Capacity v2",
        title="Shallow random forest: held-out SHAP attribution",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    colorbar = fig.colorbar(scatter_handles[-1], ax=ax, pad=0.02)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Within-feature value")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    global N_CLUSTER_BOOTSTRAPS
    args = parse_args()
    date.fromisoformat(args.run_date)
    if args.cluster_bootstraps < 1:
        raise ValueError("--cluster-bootstraps must be positive")
    N_CLUSTER_BOOTSTRAPS = args.cluster_bootstraps
    explanatory_dir = args.explanatory_dir.resolve()
    capacity_dir = args.capacity_dir.resolve()
    paths = {
        "explanatory_panel": explanatory_dir / "explanatory_panel.csv",
        "explanatory_summary": explanatory_dir / "summary.json",
        "explanatory_manifest": explanatory_dir / "manifest.json",
        "feature_dictionary": explanatory_dir / "feature_dictionary.csv",
        "capacity_panel": capacity_dir / "capacity_v2_panel.csv",
        "capacity_stability": capacity_dir / "country_stability_summary.csv",
        "capacity_weight_summary": capacity_dir / "weight_sweep_summary.csv",
        "official_outcomes": args.outcomes_file.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "results" / "generated" / args.run_date
    )
    figure_dir = (
        args.figure_dir.resolve()
        if args.figure_dir is not None
        else PROJECT_ROOT / "figures" / "generated" / args.run_date
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite revised core output: {output_dir}")

    panel = pd.read_csv(paths["explanatory_panel"])
    features = validate_explanatory_gate_and_manifest(
        panel,
        paths["explanatory_panel"],
        paths["explanatory_summary"],
        paths["explanatory_manifest"],
    )
    capacity = pd.read_csv(paths["capacity_panel"])
    stability_input = pd.read_csv(paths["capacity_stability"])
    weight_summary = pd.read_csv(paths["capacity_weight_summary"])
    outcomes = pd.read_csv(paths["official_outcomes"])
    required_panel = {"iso3", "country", "year", "legacy_category", TARGET, *features}
    if missing := sorted(required_panel - set(panel.columns)):
        raise ValueError(f"Explanatory panel lacks required columns: {missing}")
    if len(panel) != 360 or panel["iso3"].nunique() != 30:
        raise ValueError("Model panel is not the expected 30-country x 12-year scope")
    if panel.duplicated(["iso3", "year"]).any() or panel[TARGET].isna().any():
        raise ValueError("Model panel keys or target are incomplete")
    if any(panel.groupby("iso3")[feature].count().min() < 8 for feature in features):
        raise ValueError("A retained feature violates the minimum eight-years-per-country gate")

    country_summary = build_country_index_summary(capacity, stability_input)
    robustness, category_constraints = index_robustness(capacity, weight_summary)
    (
        membership,
        silhouette,
        cluster_stability,
        cluster_tree,
        cluster_scaled,
        cluster_gate,
    ) = country_structure(panel, features)
    (
        oof,
        model_metrics,
        fold_metrics,
        fold_countries,
        linear_coefficients,
        coefficient_stability,
        rf_fold_importance,
        rf_missingness_audit,
        explanation_stability,
        rf_rank_pairwise,
        cross_fitted_shap,
    ) = run_models(panel, features)
    alignment, alignment_detail, alignment_source_rows = external_alignment(
        capacity, outcomes
    )

    figure_paths = {
        "method_flow": figure_dir / "figure_01_method_flow.png",
        "capacity_profile": figure_dir / "figure_02_capacity_profile.png",
        "country_structure": figure_dir
        / (
            "figure_03_country_structure.png"
            if cluster_gate
            else "appendix_country_structure_unstable.png"
        ),
        "linear_coefficients": figure_dir / "figure_04_linear_coefficients.png",
        "rf_shap_beeswarm": figure_dir / "figure_05_rf_shap_beeswarm.png",
    }
    possible_structure_paths = [
        figure_dir / "figure_03_country_structure.png",
        figure_dir / "appendix_country_structure_unstable.png",
    ]
    existing_figures = [
        path for path in [*figure_paths.values(), *possible_structure_paths] if path.exists()
    ]
    if existing_figures:
        raise FileExistsError(
            "Refusing to overwrite or mix report figures from another run: "
            + "; ".join(map(str, sorted(set(existing_figures), key=str)))
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "country_index_summary": output_dir / "country_index_summary.csv",
        "index_robustness": output_dir / "index_robustness_summary.csv",
        "category_constraints": output_dir / "category_constraint_summary.csv",
        "cluster_membership": output_dir / "cluster_membership.csv",
        "cluster_silhouette": output_dir / "cluster_silhouette.csv",
        "cluster_stability": output_dir / "cluster_stability.csv",
        "model_oof_predictions": output_dir / "model_oof_predictions.csv",
        "model_metrics": output_dir / "model_metrics.csv",
        "fold_metrics": output_dir / "model_fold_metrics.csv",
        "fold_countries": output_dir / "model_fold_countries.csv",
        "linear_coefficients": output_dir / "linear_coefficients.csv",
        "linear_coefficient_stability": output_dir / "linear_coefficient_stability.csv",
        "rf_permutation_importance": output_dir / "rf_permutation_importance.csv",
        "rf_missingness_indicator_audit": output_dir
        / "rf_missingness_indicator_audit.csv",
        "rf_explanation_stability": output_dir / "rf_explanation_stability.csv",
        "rf_rank_pairwise": output_dir / "rf_rank_pairwise.csv",
        "rf_cross_fitted_shap": output_dir / "rf_cross_fitted_shap_values.csv",
        "external_alignment": output_dir / "external_alignment.csv",
        "external_alignment_detail": output_dir / "external_alignment_country_detail.csv",
        "external_alignment_source_rows": output_dir / "external_alignment_source_rows.csv",
    }
    frames = {
        "country_index_summary": country_summary,
        "index_robustness": robustness,
        "category_constraints": category_constraints,
        "cluster_membership": membership,
        "cluster_silhouette": silhouette,
        "cluster_stability": cluster_stability,
        "model_oof_predictions": oof,
        "model_metrics": model_metrics,
        "fold_metrics": fold_metrics,
        "fold_countries": fold_countries,
        "linear_coefficients": linear_coefficients,
        "linear_coefficient_stability": coefficient_stability,
        "rf_permutation_importance": rf_fold_importance,
        "rf_missingness_indicator_audit": rf_missingness_audit,
        "rf_explanation_stability": explanation_stability,
        "rf_rank_pairwise": rf_rank_pairwise,
        "rf_cross_fitted_shap": cross_fitted_shap,
        "external_alignment": alignment,
        "external_alignment_detail": alignment_detail,
        "external_alignment_source_rows": alignment_source_rows,
    }
    for name, frame in frames.items():
        write_csv(frame, output_paths[name])

    plot_method_flow(figure_paths["method_flow"])
    plot_capacity_profile(country_summary, figure_paths["capacity_profile"])
    plot_country_structure(
        membership,
        cluster_tree,
        cluster_scaled,
        features,
        figure_paths["country_structure"],
        cluster_gate,
    )
    plot_linear_coefficients(linear_coefficients, figure_paths["linear_coefficients"])
    plot_shap_beeswarm(cross_fitted_shap, figure_paths["rf_shap_beeswarm"])

    selected_k = int(silhouette.loc[silhouette["silhouette_score"].idxmax(), "k"])
    median_ari = float(cluster_stability["adjusted_rand_index_vs_full"].median())
    stable_features = explanation_stability.loc[
        explanation_stability["stable_for_main_text"], "feature"
    ].tolist()
    guardrails = {
        "target": TARGET,
        "target_is_constructed_composite": True,
        "target_ingredients_excluded_from_predictors": True,
        "outcomes_excluded_from_predictors": True,
        "predictor_features": features,
        "country_grouped_folds": True,
        "maximum_country_overlap_between_train_and_test": int(
            fold_countries["country_overlap_count"].max()
        ),
        "models_run": ["OLS", "fixed shallow Random Forest"],
        "fixed_missingness_indicator_set": sorted(
            rf_missingness_audit["feature"].unique().tolist()
        )
        if len(rf_missingness_audit)
        else [],
        "value_and_missingness_indicator_joint_permutation": True,
        "hyperparameter_tuning_performed": False,
        "prohibited_methods_run": [],
        "shap_outputs": ["one cross-fitted global beeswarm", "fold stability audit"],
        "ice_or_pdp_run": False,
        "forecast_run": False,
        "causal_identification_attempted": False,
        "interpretation_limit": (
            "High out-of-fold fit can reflect reconstruction of a constructed rank target; "
            "model attribution is associative and not evidence of intervention effects."
        ),
    }
    guardrail_path = output_dir / "guardrail_audit.json"
    write_json(guardrail_path, guardrails)

    summary = {
        "artifact": "minimum result package for the concise revised Food Security report",
        "scope": {
            "countries": int(panel["iso3"].nunique()),
            "model_years": f"{int(panel['year'].min())}-{int(panel['year'].max())}",
            "model_rows": len(panel),
            "index_years": f"{int(capacity['year'].min())}-{int(capacity['year'].max())}",
            "index_rows": len(capacity),
            "features_retained_after_data_gate": features,
        },
        "country_structure": {
            "selected_k_by_maximum_silhouette": selected_k,
            "silhouette_scores": silhouette.to_dict(orient="records"),
            "bootstrap_replicates": N_CLUSTER_BOOTSTRAPS,
            "median_bootstrap_adjusted_rand": median_ari,
            "main_text_stability_gate_threshold": 0.60,
            "main_text_stability_gate_pass": cluster_gate,
            "figure_role": "main" if cluster_gate else "appendix",
        },
        "model_validation": model_metrics.to_dict(orient="records"),
        "linear_attribution": {
            "standardization": "full-panel coefficients per predictor SD",
            "uncertainty": "country-clustered standard errors and 95% intervals",
            "fold_sign_stability": coefficient_stability.to_dict(orient="records"),
        },
        "random_forest_attribution": {
            "configuration": {
                "n_estimators": 500,
                "max_depth": 5,
                "min_samples_leaf": 5,
                "max_features": 0.7,
                "random_state": RANDOM_STATE,
            },
            "stable_features_for_main_text": stable_features,
            "feature_stability": explanation_stability.to_dict(orient="records"),
            "missingness_indicator_audit": rf_missingness_audit.to_dict(
                orient="records"
            ),
            "median_pairwise_fold_rank_spearman": float(
                rf_rank_pairwise["spearman_feature_rank"].median()
            ),
            "median_pairwise_top_five_jaccard": float(
                rf_rank_pairwise["top_five_jaccard"].median()
            ),
        },
        "external_alignment": alignment.to_dict(orient="records"),
        "index_robustness": robustness.to_dict(orient="records"),
        "expert_hypotheses": category_constraints.to_dict(orient="records"),
        "guardrails": guardrails,
        "inputs": {name: input_record(path) for name, path in paths.items()},
        "validation": {
            "oof_rows": len(oof),
            "oof_missing_predictions": int(
                oof[["ols_prediction", "shallow_random_forest_prediction"]]
                .isna()
                .sum()
                .sum()
            ),
            "group_fold_country_overlap": int(
                fold_countries["country_overlap_count"].sum()
            ),
            "cluster_bootstrap_rows": len(cluster_stability),
            "shap_rows": len(cross_fitted_shap),
            "shap_expected_base_feature_rows": len(panel) * len(features),
            "shap_base_feature_rows": int(
                cross_fitted_shap["feature"].isin(features).sum()
            ),
            "fixed_missingness_indicator_fold_rows": len(rf_missingness_audit),
            "main_report_models": 2,
            "explanation_stability_evaluated": True,
            "complete_method_guardrail_audit": True,
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    manifest = {
        "script": {
            "path": portable_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
        },
        "inputs": summary["inputs"],
        "outputs": [
            {
                "path": portable_path(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [
                *output_paths.values(),
                guardrail_path,
                summary_path,
                *figure_paths.values(),
            ]
        ],
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print(
        json.dumps(
            json_safe(
                {
                    "output_dir": str(output_dir),
                    "features": features,
                    "cluster": {
                        "selected_k": selected_k,
                        "median_bootstrap_ari": median_ari,
                        "main_text_gate_pass": cluster_gate,
                    },
                    "model_metrics": model_metrics.to_dict(orient="records"),
                    "stable_rf_features": stable_features,
                    "external_alignment": alignment.loc[
                        alignment["report_role"].eq("main"),
                        [
                            "outcome_view",
                            "country_count",
                            "country_median_spearman",
                            "direction_consistent",
                        ],
                    ].to_dict(orient="records"),
                    "figures": {name: str(path) for name, path in figure_paths.items()},
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
