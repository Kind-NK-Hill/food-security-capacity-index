from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

YEARS = list(range(2010, 2024))
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
LEGACY_SSR_WEIGHT = 19.0 / 34.0
LEGACY_PRODUCTION_WEIGHT = 15.0 / 34.0

CATEGORY_1 = "category_1"
CATEGORY_2 = "category_2"
CATEGORY_3 = "category_3"
FLEXIBLE_1 = "flexible_1"
FLEXIBLE_2 = "flexible_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Capacity Score v2 from processed FAOSTAT and WDI inputs."
    )
    parser.add_argument(
        "--ssr-input",
        type=Path,
        required=True,
        help="Processed FAOSTAT cereal SSR CSV produced by download_official_outcomes.py.",
    )
    parser.add_argument(
        "--wdi-input",
        type=Path,
        required=True,
        help="Processed WDI long CSV containing AG.PRD.CREL.MT and SP.POP.TOTL.",
    )
    parser.add_argument(
        "--categories-input",
        type=Path,
        default=ROOT / "config" / "country_categories.csv",
        help="CSV with iso3 and legacy_category columns.",
    )
    parser.add_argument(
        "--protocol-input",
        type=Path,
        default=ROOT / "docs" / "capacity_index_protocol.md",
        help="Method protocol included in the provenance manifest.",
    )
    parser.add_argument(
        "--run-date",
        default=date.today().isoformat(),
        help="Output label in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/generated/capacity/<run-date>.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def percentile_rank(series: pd.Series) -> pd.Series:
    if series.notna().sum() < 2:
        raise ValueError("Percentile rank requires at least two observations")
    ranks = series.rank(method="average", ascending=True)
    return (ranks - 1.0) / (series.notna().sum() - 1.0)


def minmax(series: pd.Series) -> pd.Series:
    lower = float(series.min())
    upper = float(series.max())
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Annual min-max normalization requires a finite range")
    return (series - lower) / (upper - lower)


def rank_descending(series: pd.Series) -> pd.Series:
    return series.rank(method="average", ascending=False)


def spearman_from_ranks(left: pd.Series, right: pd.Series) -> float:
    value = float(left.corr(right, method="pearson"))
    if not np.isfinite(value):
        raise ValueError("Non-finite rank correlation")
    return value


def category_constraint_values(country_scores: pd.DataFrame) -> dict[str, object]:
    grouped = {
        label: country_scores.loc[
            country_scores["legacy_category"].eq(label), "country_median_score"
        ]
        for label in [CATEGORY_1, CATEGORY_2, CATEGORY_3, FLEXIBLE_1, FLEXIBLE_2]
    }
    if any(values.empty for values in grouped.values()):
        raise ValueError("Every expert category must have at least one modern-v2 country")

    margins = {
        "margin_min_category1_minus_max_category2": float(
            grouped[CATEGORY_1].min() - grouped[CATEGORY_2].max()
        ),
        "margin_min_category2_minus_max_category3": float(
            grouped[CATEGORY_2].min() - grouped[CATEGORY_3].max()
        ),
        "margin_min_flexible1_minus_max_category3": float(
            grouped[FLEXIBLE_1].min() - grouped[CATEGORY_3].max()
        ),
        "margin_min_category1_minus_max_flexible2": float(
            grouped[CATEGORY_1].min() - grouped[FLEXIBLE_2].max()
        ),
    }
    return {
        **margins,
        "all_category_constraints_pass": all(value > 0.0 for value in margins.values()),
    }


def contiguous_intervals(feasible_weights: list[float]) -> list[dict[str, float]]:
    if not feasible_weights:
        return []
    intervals: list[dict[str, float]] = []
    start = previous = feasible_weights[0]
    for weight in feasible_weights[1:]:
        if round(weight - previous, 2) != 0.01:
            intervals.append({"start": float(start), "end": float(previous)})
            start = weight
        previous = weight
    intervals.append({"start": float(start), "end": float(previous)})
    return intervals


def main() -> None:
    args = parse_args()
    date.fromisoformat(args.run_date)
    ssr_input = args.ssr_input.resolve()
    wdi_input = args.wdi_input.resolve()
    categories_input = args.categories_input.resolve()
    protocol_input = args.protocol_input.resolve()
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT / "results" / "generated" / "capacity" / args.run_date
    )
    for label, path in {
        "SSR input": ssr_input,
        "WDI input": wdi_input,
        "categories input": categories_input,
        "protocol input": protocol_input,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {out_dir}")
    out_dir.mkdir(parents=True)

    ssr_raw = pd.read_csv(ssr_input, dtype={"iso3": str})
    ssr = ssr_raw.loc[
        ssr_raw["value_status"].eq("observed_components")
        & ssr_raw["year"].isin(YEARS),
        [
            "iso3",
            "country",
            "year",
            "production_1000_tonnes",
            "import_1000_tonnes",
            "export_1000_tonnes",
            "fbs_cereal_ssr_candidate_pct",
            "production_flag",
            "import_flag",
            "export_flag",
            "source_url",
            "accessed_at_utc",
        ],
    ].copy()
    if len(ssr) != 420 or ssr["iso3"].nunique() != 30:
        raise ValueError("Expected 420 observed SSR rows for 30 countries")
    if "JPN" in set(ssr["iso3"]):
        raise ValueError("Japan must remain absent from the modern SSR sample")
    if ssr.duplicated(["iso3", "year"]).any():
        raise ValueError("Duplicate SSR country-year keys")

    wdi_raw = pd.read_csv(wdi_input, dtype={"iso3": str})
    wdi = wdi_raw.loc[
        wdi_raw["indicator_code"].isin(["AG.PRD.CREL.MT", "SP.POP.TOTL"])
        & wdi_raw["year"].isin(YEARS),
        ["iso3", "year", "indicator_code", "value", "wdi_lastupdated", "api_url", "accessed_at_utc"],
    ].copy()
    wdi["value"] = pd.to_numeric(wdi["value"], errors="coerce")
    if wdi.duplicated(["iso3", "year", "indicator_code"]).any():
        raise ValueError("Duplicate WDI country-year-indicator keys")
    wdi_values = wdi.pivot(index=["iso3", "year"], columns="indicator_code", values="value").reset_index()
    wdi_values = wdi_values.rename(
        columns={
            "AG.PRD.CREL.MT": "wdi_cereal_production_metric_tonnes",
            "SP.POP.TOTL": "wdi_population_persons",
        }
    )
    wdi_meta = (
        wdi.groupby(["iso3", "year"], as_index=False)
        .agg(
            wdi_lastupdated=("wdi_lastupdated", "first"),
            wdi_accessed_at_utc=("accessed_at_utc", "first"),
        )
    )
    wdi_values = wdi_values.merge(wdi_meta, on=["iso3", "year"], validate="one_to_one")

    categories_raw = pd.read_csv(categories_input, dtype={"iso3": str})
    required_category_columns = {"iso3", "legacy_category"}
    if missing := sorted(required_category_columns - set(categories_raw.columns)):
        raise ValueError(f"Category file lacks required columns: {missing}")
    category_counts = categories_raw.groupby("iso3")["legacy_category"].nunique(dropna=False)
    if not category_counts.eq(1).all():
        raise ValueError("Legacy category is not stable within country")
    categories = categories_raw[["iso3", "legacy_category"]].drop_duplicates("iso3")

    panel = ssr.merge(wdi_values, on=["iso3", "year"], how="left", validate="one_to_one")
    panel = panel.merge(categories, on="iso3", how="left", validate="many_to_one")
    required_numeric = [
        "fbs_cereal_ssr_candidate_pct",
        "wdi_cereal_production_metric_tonnes",
        "wdi_population_persons",
    ]
    if panel[required_numeric].isna().any().any():
        raise ValueError("Missing modern-v2 component values after merge")
    if (panel["wdi_population_persons"] <= 0).any():
        raise ValueError("Population must be positive")
    panel["wdi_cereal_production_tonnes_per_person"] = (
        panel["wdi_cereal_production_metric_tonnes"] / panel["wdi_population_persons"]
    )

    panel["ssr_percentile_rank"] = panel.groupby("year")[
        "fbs_cereal_ssr_candidate_pct"
    ].transform(percentile_rank)
    panel["production_per_person_percentile_rank"] = panel.groupby("year")[
        "wdi_cereal_production_tonnes_per_person"
    ].transform(percentile_rank)
    panel["ssr_annual_minmax"] = panel.groupby("year")[
        "fbs_cereal_ssr_candidate_pct"
    ].transform(minmax)
    panel["production_per_person_annual_minmax"] = panel.groupby("year")[
        "wdi_cereal_production_tonnes_per_person"
    ].transform(minmax)

    panel["capacity_v2_rank_50_50"] = 0.5 * panel["ssr_percentile_rank"] + 0.5 * panel[
        "production_per_person_percentile_rank"
    ]
    panel["capacity_v2_rank_legacy_56_44"] = (
        LEGACY_SSR_WEIGHT * panel["ssr_percentile_rank"]
        + LEGACY_PRODUCTION_WEIGHT * panel["production_per_person_percentile_rank"]
    )
    panel["capacity_v2_minmax_50_50"] = 0.5 * panel["ssr_annual_minmax"] + 0.5 * panel[
        "production_per_person_annual_minmax"
    ]
    panel["annual_rank_capacity_v2_50_50"] = panel.groupby("year")[
        "capacity_v2_rank_50_50"
    ].transform(rank_descending)

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    panel_output = out_dir / "capacity_v2_panel.csv"
    panel.to_csv(panel_output, index=False)

    primary_country = (
        panel.groupby(["iso3", "country", "legacy_category"], as_index=False, dropna=False)
        .agg(primary_country_median_score=("capacity_v2_rank_50_50", "median"))
    )
    primary_country["primary_country_rank"] = rank_descending(
        primary_country["primary_country_median_score"]
    )
    reference_ranks = primary_country.set_index("iso3")["primary_country_rank"]

    sweep_country_frames: list[pd.DataFrame] = []
    sweep_summary_rows: list[dict[str, object]] = []
    for weight in WEIGHTS:
        weighted = panel.assign(
            sweep_score=(
                float(weight) * panel["ssr_percentile_rank"]
                + (1.0 - float(weight)) * panel["production_per_person_percentile_rank"]
            )
        )
        country_scores = (
            weighted.groupby(["iso3", "country", "legacy_category"], as_index=False, dropna=False)
            .agg(country_median_score=("sweep_score", "median"))
        )
        country_scores["country_rank"] = rank_descending(country_scores["country_median_score"])
        country_scores["ssr_weight"] = float(weight)
        country_scores["production_per_person_weight"] = 1.0 - float(weight)
        country_scores["top_five"] = country_scores["country_rank"] <= 5
        country_scores["bottom_five"] = country_scores["country_rank"] >= 26
        sweep_country_frames.append(country_scores)

        aligned_reference = country_scores["iso3"].map(reference_ranks)
        constraints = category_constraint_values(country_scores)
        sweep_summary_rows.append(
            {
                "ssr_weight": float(weight),
                "production_per_person_weight": 1.0 - float(weight),
                "spearman_country_rank_vs_50_50": spearman_from_ranks(
                    country_scores["country_rank"], aligned_reference
                ),
                **constraints,
            }
        )

    sweep_country = pd.concat(sweep_country_frames, ignore_index=True)
    sweep_country = sweep_country.sort_values(["ssr_weight", "country_rank", "iso3"]).reset_index(drop=True)
    sweep_summary = pd.DataFrame(sweep_summary_rows).sort_values("ssr_weight").reset_index(drop=True)
    sweep_country_output = out_dir / "weight_sweep_country_ranks.csv"
    sweep_summary_output = out_dir / "weight_sweep_summary.csv"
    sweep_country.to_csv(sweep_country_output, index=False)
    sweep_summary.to_csv(sweep_summary_output, index=False)

    minmax_country = (
        panel.groupby(["iso3", "country"], as_index=False)
        .agg(minmax_country_median_score=("capacity_v2_minmax_50_50", "median"))
    )
    minmax_country["minmax_country_rank"] = rank_descending(
        minmax_country["minmax_country_median_score"]
    )
    legacy_weight_country = (
        panel.groupby(["iso3", "country"], as_index=False)
        .agg(legacy_56_44_country_median_score=("capacity_v2_rank_legacy_56_44", "median"))
    )
    legacy_weight_country["legacy_56_44_country_rank"] = rank_descending(
        legacy_weight_country["legacy_56_44_country_median_score"]
    )

    temporal_rows: list[dict[str, object]] = []
    for (iso3, country, legacy_category), group in panel.groupby(
        ["iso3", "country", "legacy_category"], dropna=False, sort=True
    ):
        ordered = group.sort_values("year")
        temporal_rows.append(
            {
                "iso3": iso3,
                "country": country,
                "legacy_category": legacy_category,
                "primary_annual_score_sd_population": float(
                    ordered["capacity_v2_rank_50_50"].std(ddof=0)
                ),
                "primary_mean_absolute_annual_change": float(
                    ordered["capacity_v2_rank_50_50"].diff().abs().dropna().mean()
                ),
                "ssr_percentile_rank_sd_population": float(
                    ordered["ssr_percentile_rank"].std(ddof=0)
                ),
                "production_per_person_rank_sd_population": float(
                    ordered["production_per_person_percentile_rank"].std(ddof=0)
                ),
            }
        )
    stability = pd.DataFrame(temporal_rows)
    all_weight_stability = (
        sweep_country.groupby("iso3", as_index=False)
        .agg(
            all_weight_rank_min=("country_rank", "min"),
            all_weight_rank_max=("country_rank", "max"),
            all_weight_top5_share=("top_five", "mean"),
            all_weight_bottom5_share=("bottom_five", "mean"),
        )
    )
    all_weight_stability["all_weight_rank_range"] = (
        all_weight_stability["all_weight_rank_max"] - all_weight_stability["all_weight_rank_min"]
    )
    central_sweep = sweep_country.loc[sweep_country["ssr_weight"].between(0.25, 0.75)]
    central_weight_stability = (
        central_sweep.groupby("iso3", as_index=False)
        .agg(
            central_weight_rank_min=("country_rank", "min"),
            central_weight_rank_max=("country_rank", "max"),
            central_weight_top5_share=("top_five", "mean"),
            central_weight_bottom5_share=("bottom_five", "mean"),
        )
    )
    central_weight_stability["central_weight_rank_range"] = (
        central_weight_stability["central_weight_rank_max"]
        - central_weight_stability["central_weight_rank_min"]
    )
    stability = stability.merge(primary_country, on=["iso3", "country", "legacy_category"], validate="one_to_one")
    stability = stability.merge(legacy_weight_country, on=["iso3", "country"], validate="one_to_one")
    stability = stability.merge(minmax_country, on=["iso3", "country"], validate="one_to_one")
    stability = stability.merge(all_weight_stability, on="iso3", validate="one_to_one")
    stability = stability.merge(central_weight_stability, on="iso3", validate="one_to_one")
    stability = stability.sort_values(["primary_country_rank", "iso3"]).reset_index(drop=True)
    stability_output = out_dir / "country_stability_summary.csv"
    stability.to_csv(stability_output, index=False)

    correlation_rows: list[dict[str, object]] = []
    for year, group in panel.groupby("year", sort=True):
        correlation_rows.append(
            {
                "scope": "annual_cross_section",
                "year": int(year),
                "countries": int(len(group)),
                "spearman_ssr_vs_production_per_person": spearman_from_ranks(
                    group["ssr_percentile_rank"], group["production_per_person_percentile_rank"]
                ),
            }
        )
    component_country = (
        panel.groupby("iso3", as_index=False)
        .agg(
            median_ssr_rank=("ssr_percentile_rank", "median"),
            median_production_per_person_rank=("production_per_person_percentile_rank", "median"),
        )
    )
    correlation_rows.append(
        {
            "scope": "country_median_2010_2023",
            "year": pd.NA,
            "countries": int(len(component_country)),
            "spearman_ssr_vs_production_per_person": spearman_from_ranks(
                component_country["median_ssr_rank"].rank(method="average"),
                component_country["median_production_per_person_rank"].rank(method="average"),
            ),
        }
    )
    component_correlation = pd.DataFrame(correlation_rows)
    correlation_output = out_dir / "component_rank_correlation.csv"
    component_correlation.to_csv(correlation_output, index=False)

    feasible_weights = sweep_summary.loc[
        sweep_summary["all_category_constraints_pass"], "ssr_weight"
    ].astype(float).tolist()
    central_summary = sweep_summary.loc[sweep_summary["ssr_weight"].between(0.25, 0.75)]
    normalization_rank_correlation = spearman_from_ranks(
        primary_country.sort_values("iso3")["primary_country_rank"].reset_index(drop=True),
        minmax_country.sort_values("iso3")["minmax_country_rank"].reset_index(drop=True),
    )
    legacy_weight_rank_correlation = spearman_from_ranks(
        primary_country.sort_values("iso3")["primary_country_rank"].reset_index(drop=True),
        legacy_weight_country.sort_values("iso3")["legacy_56_44_country_rank"].reset_index(drop=True),
    )
    central_min_agreement = float(central_summary["spearman_country_rank_vs_50_50"].min())
    weight_50_feasible = bool(
        sweep_summary.loc[sweep_summary["ssr_weight"].eq(0.5), "all_category_constraints_pass"].iloc[0]
    )
    if (
        weight_50_feasible
        and central_min_agreement >= 0.90
        and normalization_rank_correlation >= 0.90
    ):
        verdict = "construct_index_supported"
    elif not feasible_weights and central_min_agreement < 0.70:
        verdict = "reconsider_index"
    else:
        verdict = "construct_index_with_caution"

    score_formula_error = float(
        (
            panel["capacity_v2_rank_50_50"]
            - 0.5
            * (panel["ssr_percentile_rank"] + panel["production_per_person_percentile_rank"])
        )
        .abs()
        .max()
    )
    validation = {
        "panel_rows": int(len(panel)),
        "unique_iso3_year_keys": int(panel[["iso3", "year"]].drop_duplicates().shape[0]),
        "countries": int(panel["iso3"].nunique()),
        "years": int(panel["year"].nunique()),
        "japan_rows": int(panel["iso3"].eq("JPN").sum()),
        "component_missing_cells": int(panel[required_numeric].isna().sum().sum()),
        "score_formula_max_absolute_error": score_formula_error,
        "primary_scores_within_zero_one": bool(panel["capacity_v2_rank_50_50"].between(0, 1).all()),
        "weight_grid_rows": int(len(sweep_summary)),
        "weight_country_rows": int(len(sweep_country)),
        "unique_weight_country_keys": int(
            sweep_country[["ssr_weight", "iso3"]].drop_duplicates().shape[0]
        ),
        "stability_rows": int(len(stability)),
        "outcome_columns_used": [],
        "point_imputation": False,
        "predictive_model_run": False,
        "tree_or_shap_run": False,
    }
    if validation != {
        **validation,
        "panel_rows": 420,
        "unique_iso3_year_keys": 420,
        "countries": 30,
        "years": 14,
        "japan_rows": 0,
        "component_missing_cells": 0,
        "weight_grid_rows": 101,
        "weight_country_rows": 3030,
        "unique_weight_country_keys": 3030,
        "stability_rows": 30,
    }:
        raise ValueError("Unexpected validation counts")
    if score_formula_error > 1e-12 or not validation["primary_scores_within_zero_one"]:
        raise ValueError("Primary score validation failed")

    summary = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": "2026-08-29",
            "verification_status": "UNVERIFIED",
            "version_label": "exp_result_v1",
        },
        "scope": {
            "countries": 30,
            "years": [2010, 2023],
            "balanced_panel_rows": 420,
            "japan_treatment": "absent_from_modern_v2_not_imputed_retained_in_legacy_comparison",
            "construct": "cereal_supply_trade_capacity_not_total_food_security",
        },
        "primary_specification": {
            "normalization": "annual_cross_sectional_percentile_rank_0_1",
            "ssr_weight": 0.5,
            "production_per_person_weight": 0.5,
            "stability_in_score": False,
        },
        "weight_sensitivity": {
            "grid_start": 0.0,
            "grid_end": 1.0,
            "grid_step": 0.01,
            "feasible_weight_count": len(feasible_weights),
            "feasible_intervals": contiguous_intervals(feasible_weights),
            "weight_50_category_feasible": weight_50_feasible,
            "central_025_075_min_spearman_vs_50_50": central_min_agreement,
            "legacy_56_44_spearman_vs_50_50": legacy_weight_rank_correlation,
        },
        "normalization_sensitivity": {
            "rank_vs_annual_minmax_country_order_spearman": normalization_rank_correlation,
        },
        "component_relationship": {
            "country_median_ssr_vs_production_per_person_spearman": float(
                component_correlation.loc[
                    component_correlation["scope"].eq("country_median_2010_2023"),
                    "spearman_ssr_vs_production_per_person",
                ].iloc[0]
            )
        },
        "stability_diagnostic": {
            "median_primary_annual_score_sd": float(
                stability["primary_annual_score_sd_population"].median()
            ),
            "median_primary_mean_absolute_annual_change": float(
                stability["primary_mean_absolute_annual_change"].median()
            ),
            "median_central_weight_rank_range": float(
                stability["central_weight_rank_range"].median()
            ),
            "maximum_central_weight_rank_range": float(
                stability["central_weight_rank_range"].max()
            ),
        },
        "scope_verdict": verdict,
        "validation": validation,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    summary_output = out_dir / "summary.json"
    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    output_paths = [
        panel_output,
        sweep_country_output,
        sweep_summary_output,
        stability_output,
        correlation_output,
        summary_output,
    ]
    manifest = {
        "script": file_record(Path(__file__).resolve()),
        "inputs": {
            ssr_input.name: file_record(ssr_input),
            wdi_input.name: file_record(wdi_input),
            categories_input.name: file_record(categories_input),
            protocol_input.name: file_record(protocol_input),
        },
        "outputs": [file_record(path) for path in output_paths],
    }
    manifest_output = out_dir / "manifest.json"
    manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Balanced panel: {len(panel)} rows, {panel['iso3'].nunique()} countries, {panel['year'].nunique()} years")
    print(f"Category-feasible weights: {len(feasible_weights)} / {len(WEIGHTS)}")
    print(f"Feasible intervals: {contiguous_intervals(feasible_weights)}")
    print(f"Central weight minimum Spearman vs 50/50: {central_min_agreement:.6f}")
    print(f"Rank vs min-max country-order Spearman: {normalization_rank_correlation:.6f}")
    print(f"Scope verdict: {verdict}")


if __name__ == "__main__":
    main()
