#!/usr/bin/env python3
"""Reproduce the original Legacy three-weight grid as an appendix audit.

The preserved Rmd called this a Monte Carlo optimisation.  It is instead the
complete deterministic 1% simplex: 5,151 non-negative weight triples summing
to one.  This script reproduces the four category constraints, the original
10-bin TV diagnostic, traversal-order tie behaviour, and rank sensitivity.
It does not endorse the Legacy GHI construction as the revised primary index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PANEL_ROWS = 31 * 28
EXPECTED_GRID_ROWS = 5151
ORIGINAL_FIXED_WEIGHTS = (0.19, 0.15, 0.66)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the complete deterministic Legacy 1% weight simplex."
    )
    parser.add_argument(
        "--run-date",
        default=date.today().isoformat(),
        help="Output folder label in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--input-panel",
        type=Path,
        required=True,
        help="Legacy 31-country panel containing the three normalized components and fixed score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/generated/legacy/<run-date>.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=None,
        help="Simplex figure path. Defaults inside the output directory.",
    )
    return parser.parse_args()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def original_r_histogram_tv(values: np.ndarray, bins: int = 10) -> float:
    """Match R hist(..., right=TRUE, include.lowest=TRUE) for the Rmd rule."""
    lower = min(0.15, float(np.min(values)))
    upper = max(0.75, float(np.max(values)))
    edges = np.linspace(lower, upper, bins + 1)
    # R's default intervals are [lowest, b1], (b1, b2], ..., (b9, highest].
    memberships = np.digitize(values, edges[1:-1], right=True)
    counts = np.bincount(memberships, minlength=bins)
    proportions = counts / counts.sum()
    return float(0.5 * np.abs(proportions - 1.0 / bins).sum())


def simplex_grid() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    traversal = 0
    # expand.grid(w1, w2) in R varies w1 fastest, then filters w3 >= 0.
    for w2_integer in range(101):
        for w1_integer in range(101 - w2_integer):
            traversal += 1
            w3_integer = 100 - w1_integer - w2_integer
            records.append(
                {
                    "traversal_order": traversal,
                    "w1_index_1": w1_integer / 100.0,
                    "w2_index_2": w2_integer / 100.0,
                    "w3_one_minus_ghi": w3_integer / 100.0,
                }
            )
    grid = pd.DataFrame(records)
    if len(grid) != EXPECTED_GRID_ROWS:
        raise AssertionError(f"Simplex has {len(grid)} rows, expected {EXPECTED_GRID_ROWS}")
    return grid


def category_indices(country_table: pd.DataFrame, category: str) -> np.ndarray:
    indices = np.flatnonzero(country_table["legacy_category"].eq(category).to_numpy())
    if len(indices) == 0:
        raise ValueError(f"Legacy category is empty: {category}")
    return indices


def rank_columns_descending(values: np.ndarray) -> np.ndarray:
    """Average descending ranks for each grid column."""
    return pd.DataFrame(values).rank(axis=0, method="average", ascending=False).to_numpy()


def main() -> int:
    args = parse_args()
    date.fromisoformat(args.run_date)
    panel_path = args.input_panel.resolve()
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing Legacy panel: {panel_path}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "results" / "generated" / "legacy" / args.run_date
    )
    figure_path = (
        args.figure_path.resolve()
        if args.figure_path is not None
        else output_dir / "legacy_simplex.png"
    )
    figure_dir = figure_path.parent
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Legacy grid output: {output_dir}")
    if figure_path.exists():
        raise FileExistsError(f"Refusing to overwrite Legacy simplex figure: {figure_path}")

    panel = pd.read_csv(panel_path).sort_values(["iso3", "year"], ignore_index=True)
    required = {
        "iso3",
        "country",
        "year",
        "legacy_category",
        "index_1_norm_legacy_global",
        "index_2_norm_legacy_global",
        "ghi_norm_legacy_global",
        "new_mixed_index_fixed_legacy_weights",
    }
    missing_columns = sorted(required - set(panel.columns))
    if missing_columns:
        raise ValueError(f"Legacy panel lacks required columns: {missing_columns}")
    if len(panel) != EXPECTED_PANEL_ROWS:
        raise ValueError(f"Legacy panel has {len(panel)} rows, expected {EXPECTED_PANEL_ROWS}")
    if panel.duplicated(["iso3", "year"]).any() or panel[list(required)].isna().any().any():
        raise ValueError("Legacy panel keys or required values are incomplete")

    country_table = (
        panel[["iso3", "country", "legacy_category"]]
        .drop_duplicates()
        .sort_values("iso3", ignore_index=True)
    )
    if len(country_table) != 31:
        raise ValueError(f"Expected 31 Legacy countries, found {len(country_table)}")
    years_per_country = panel.groupby("iso3")["year"].nunique()
    if not years_per_country.eq(28).all():
        raise ValueError("Each Legacy country must have exactly 28 years")

    grid = simplex_grid()
    weights = grid[["w1_index_1", "w2_index_2", "w3_one_minus_ghi"]].to_numpy()
    components = panel[
        [
            "index_1_norm_legacy_global",
            "index_2_norm_legacy_global",
            "ghi_norm_legacy_global",
        ]
    ].to_numpy(dtype=float)
    components[:, 2] = 1.0 - components[:, 2]
    row_scores = components @ weights.T

    country_medians = np.empty((len(country_table), len(grid)), dtype=float)
    for country_index, iso3 in enumerate(country_table["iso3"]):
        rows = panel["iso3"].eq(iso3).to_numpy()
        country_medians[country_index, :] = np.median(row_scores[rows, :], axis=0)

    c1 = category_indices(country_table, "category_1")
    c2 = category_indices(country_table, "category_2")
    c3 = category_indices(country_table, "category_3")
    flex1 = category_indices(country_table, "flexible_1")
    flex2 = category_indices(country_table, "flexible_2")

    min_c1 = country_medians[c1].min(axis=0)
    max_c2 = country_medians[c2].max(axis=0)
    min_c2 = country_medians[c2].min(axis=0)
    max_c3 = country_medians[c3].max(axis=0)
    min_flex1 = country_medians[flex1].min(axis=0)
    max_flex2 = country_medians[flex2].max(axis=0)

    margin_1 = min_c1 - max_c2
    margin_2 = min_c2 - max_c3
    margin_3 = min_flex1 - max_c3
    margin_4 = min_c1 - max_flex2
    check_1 = margin_1 > 0
    check_2 = margin_2 > 0
    check_3 = margin_3 > 0
    check_4 = margin_4 > 0
    valid = check_1 & check_2 & check_3 & check_4

    tv = np.array(
        [original_r_histogram_tv(country_medians[:, column]) for column in range(len(grid))]
    )
    valid_tv = np.where(valid, tv, np.nan)
    if not valid.any():
        raise ValueError("No Legacy weight combination satisfies all four category constraints")
    minimum_valid_tv = float(np.nanmin(valid_tv))
    tied_minimum = valid & np.isclose(tv, minimum_valid_tv, rtol=0.0, atol=1e-12)
    first_best_index = int(np.flatnonzero(tied_minimum)[0])

    original_mask = (
        np.isclose(weights[:, 0], ORIGINAL_FIXED_WEIGHTS[0], atol=1e-12)
        & np.isclose(weights[:, 1], ORIGINAL_FIXED_WEIGHTS[1], atol=1e-12)
        & np.isclose(weights[:, 2], ORIGINAL_FIXED_WEIGHTS[2], atol=1e-12)
    )
    if original_mask.sum() != 1:
        raise AssertionError("The preserved 0.19/0.15/0.66 weight triple is not unique in the grid")
    original_index = int(np.flatnonzero(original_mask)[0])
    fixed_score_max_abs_difference = float(
        np.max(
            np.abs(
                row_scores[:, original_index]
                - panel["new_mixed_index_fixed_legacy_weights"].to_numpy()
            )
        )
    )
    if fixed_score_max_abs_difference > 1e-12:
        raise ValueError(
            "Legacy fixed-score reproduction failed: maximum absolute difference "
            f"{fixed_score_max_abs_difference} exceeds 1e-12"
        )

    grid = grid.assign(
        min_category_1=min_c1,
        max_category_2=max_c2,
        margin_min_category_1_minus_max_category_2=margin_1,
        constraint_category_1_gt_category_2=check_1,
        min_category_2=min_c2,
        max_category_3=max_c3,
        margin_min_category_2_minus_max_category_3=margin_2,
        constraint_category_2_gt_category_3=check_2,
        min_flexible_1=min_flex1,
        margin_min_flexible_1_minus_max_category_3=margin_3,
        constraint_flexible_1_gt_category_3=check_3,
        max_flexible_2=max_flex2,
        margin_min_category_1_minus_max_flexible_2=margin_4,
        constraint_flexible_2_lt_category_1=check_4,
        all_four_constraints_pass=valid,
        tv_distance_original_10_bin_rule=tv,
        tv_eligible_under_original_workflow=valid_tv,
        tied_minimum_tv_among_valid=tied_minimum,
        first_minimum_by_r_expand_grid_traversal=(np.arange(len(grid)) == first_best_index),
        preserved_fixed_19_15_66=original_mask,
    )

    ranks = rank_columns_descending(country_medians)
    valid_ranks = ranks[:, valid]
    country_records: list[dict[str, Any]] = []
    for country_index, country in country_table.iterrows():
        country_records.append(
            {
                "iso3": country["iso3"],
                "country": country["country"],
                "legacy_category": country["legacy_category"],
                "all_grid_rank_min": float(ranks[country_index].min()),
                "all_grid_rank_max": float(ranks[country_index].max()),
                "all_grid_rank_range": float(
                    ranks[country_index].max() - ranks[country_index].min()
                ),
                "valid_grid_rank_min": float(valid_ranks[country_index].min()),
                "valid_grid_rank_max": float(valid_ranks[country_index].max()),
                "valid_grid_rank_range": float(
                    valid_ranks[country_index].max() - valid_ranks[country_index].min()
                ),
                "rank_at_first_minimum_tv": float(ranks[country_index, first_best_index]),
                "rank_at_preserved_19_15_66": float(ranks[country_index, original_index]),
                "median_score_at_first_minimum_tv": float(
                    country_medians[country_index, first_best_index]
                ),
                "median_score_at_preserved_19_15_66": float(
                    country_medians[country_index, original_index]
                ),
            }
        )
    sensitivity = pd.DataFrame(country_records).sort_values(
        ["rank_at_preserved_19_15_66", "iso3"], ignore_index=True
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "legacy_weight_grid.csv"
    sensitivity_path = output_dir / "country_rank_sensitivity.csv"
    write_csv(grid, grid_path)
    write_csv(sensitivity, sensitivity_path)

    valid_grid = grid.loc[grid["all_four_constraints_pass"]]
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    ax.scatter(
        grid["w1_index_1"],
        grid["w2_index_2"],
        s=5,
        color="#D9D9D9",
        alpha=0.45,
        linewidths=0,
        label="Fails at least one category constraint",
    )
    scatter = ax.scatter(
        valid_grid["w1_index_1"],
        valid_grid["w2_index_2"],
        c=valid_grid["tv_distance_original_10_bin_rule"],
        cmap="viridis_r",
        s=15,
        linewidths=0,
        label="Passes all four constraints",
    )
    selected = grid.iloc[first_best_index]
    ax.scatter(
        [selected["w1_index_1"]],
        [selected["w2_index_2"]],
        marker="*",
        s=190,
        color="#C51B7D",
        edgecolor="white",
        linewidth=0.8,
        label="First tied minimum by traversal",
        zorder=5,
    )
    ax.set(
        xlabel="Legacy weight on Index 1",
        ylabel="Legacy weight on Index 2",
        title="Deterministic Legacy 1% simplex (5,151 combinations)",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Original 10-bin TV diagnostic")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    first_best = grid.iloc[first_best_index]
    original = grid.iloc[original_index]
    summary = {
        "artifact": "deterministic Legacy 1% simplex and original TV diagnostic audit",
        "method": {
            "grid": "complete non-negative integer-percent triples summing to 100",
            "grid_rows": len(grid),
            "not_monte_carlo": True,
            "country_statistic": "median Legacy composite across 1994-2021",
            "constraint_count": 4,
            "tv_rule": "Rmd 10-bin histogram versus 0.1 per bin; descriptive diagnostic only",
            "tv_not_interpreted_as": ["fairness", "validity", "external validation"],
            "selection_rule": "minimum TV among constraint-valid weights; first row wins ties",
            "traversal": "R expand.grid order with w1 varying fastest, followed by w2",
        },
        "results": {
            "valid_weight_count": int(valid.sum()),
            "minimum_valid_tv": minimum_valid_tv,
            "minimum_tv_tie_count": int(tied_minimum.sum()),
            "first_minimum_by_traversal": {
                "traversal_order": int(first_best["traversal_order"]),
                "w1": float(first_best["w1_index_1"]),
                "w2": float(first_best["w2_index_2"]),
                "w3": float(first_best["w3_one_minus_ghi"]),
                "tv": float(first_best["tv_distance_original_10_bin_rule"]),
            },
            "preserved_19_15_66": {
                "traversal_order": int(original["traversal_order"]),
                "constraints_pass": bool(original["all_four_constraints_pass"]),
                "tv": float(original["tv_distance_original_10_bin_rule"]),
                "is_tied_minimum_tv": bool(original["tied_minimum_tv_among_valid"]),
                "is_first_minimum_by_traversal": bool(
                    original["first_minimum_by_r_expand_grid_traversal"]
                ),
                "exactly_recovered_by_original_first_minimum_rule": bool(
                    original_index == first_best_index
                ),
            },
            "reproduction_gate_pass": bool(
                len(grid) == EXPECTED_GRID_ROWS
                and fixed_score_max_abs_difference <= 1e-12
                and original["all_four_constraints_pass"]
                and original["tied_minimum_tv_among_valid"]
                and original_index == first_best_index
            ),
            "tie_implication": (
                "The reported triple is traversal-order dependent because several valid triples share the minimum TV."
                if tied_minimum.sum() > 1
                else "No minimum-TV tie was found on the preserved 1% grid."
            ),
        },
        "input": {
            "path": portable_path(panel_path),
            "sha256": sha256_file(panel_path),
            "bytes": panel_path.stat().st_size,
        },
        "validation": {
            "panel_rows": len(panel),
            "countries": int(panel["iso3"].nunique()),
            "years": f"{int(panel['year'].min())}-{int(panel['year'].max())}",
            "grid_rows": len(grid),
            "weight_sums_equal_one": bool(
                np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
            ),
            "fixed_weight_panel_max_abs_difference": fixed_score_max_abs_difference,
            "model_fitted": False,
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
        "inputs": [summary["input"]],
        "outputs": [
            {
                "path": portable_path(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [grid_path, sensitivity_path, summary_path, figure_path]
        ],
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "grid_rows": len(grid),
                "valid_weight_count": int(valid.sum()),
                "minimum_tv_tie_count": int(tied_minimum.sum()),
                "first_minimum": summary["results"]["first_minimum_by_traversal"],
                "preserved_19_15_66": summary["results"]["preserved_19_15_66"],
                "figure": str(figure_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
