# Capacity Score v2: 30-Country Modern Panel Protocol

## Purpose

This public protocol records the frozen 2026 reconstruction choices. A live
data refresh must use a new output directory and report any changed coverage or
source vintage rather than silently replacing the frozen result.

## Objective

Construct and audit a transparent modern cereal-capacity index without using hunger outcomes as inputs or selecting the specification for predictive performance.

The index asks a descriptive construct question: do cereal self-sufficiency
derived from official FAOSTAT Food Balances and WDI cereal production per
person produce a reasonably stable country-capacity ordering across transparent
weights and normalizations?

## Scope

- Countries: the 30 course-project countries with complete FAOSTAT Food Balances cereal SSR data. Japan remains missing and is not imputed.
- Years: 2010–2023, yielding a balanced 30 × 14 panel.
- Legacy comparison: Japan remains in the separate 31-country Legacy v1 artifacts and is not removed from the project.
- Claim boundary: cereal supply/trade capacity, not total food security, nutrition, household access, resilience, or causal impact.

## Inputs

| Component | Source | Definition |
| --- | --- | --- |
| Cereal self-sufficiency | Saved FAOSTAT Food Balances item 2905 | `production / (production + imports - exports) × 100`; values above 100 are retained for net exporters |
| Cereal production per person | Saved WDI cereal production and population series | `AG.PRD.CREL.MT / SP.POP.TOTL`; metric tonnes per person |
| Expert category | `config/country_categories.csv` | Historical Category 1/2/3, Flexible 1/2, or unclassified labels from the original sole-author course project; diagnostic only |

PoU, FIES, GHI, and the inspected 2019 PoU test fold are prohibited from index construction and weight selection.

## Primary Construction

For each year and each component, calculate a cross-sectional percentile rank on `[0,1]`:

`(average_rank_ascending - 1) / (country_count - 1)`

The primary score is:

`Capacity v2 = 0.50 × SSR percentile rank + 0.50 × production-per-person percentile rank`

The primary country ordering uses each country's median annual primary score over 2010–2023.

## Prespecified Sensitivities

1. Legacy-derived capacity weights: `19/34` on SSR and `15/34` on production per person.
2. Annual min–max normalization with 50/50 weights.
3. Complete one-dimensional rank-normalized weight sweep: SSR weight `0.00, 0.01, ..., 1.00`; production weight is `1 - SSR weight`.
4. Category constraints are evaluated on country median scores for every weight:
   - minimum Category 1 > maximum Category 2;
   - minimum Category 2 > maximum Category 3;
   - minimum Flexible 1 > maximum Category 3;
   - minimum Category 1 > maximum Flexible 2.

The feasible interval is reported, not treated as proof that the expert categories are ground truth. Distribution-uniformity or PoU performance is not used to choose a weight.

## Stability Diagnostics

Stability is not a score component.

For each country report:

- standard deviation of annual primary scores;
- mean absolute year-to-year change in the primary score;
- standard deviation of each component's annual percentile rank;
- minimum, maximum, and range of the country-median rank over the full weight sweep;
- the same rank range over the practical central weight range `[0.25,0.75]`;
- frequency of appearing in the top or bottom five across weights.

Also report Spearman rank agreement with the 50/50 country ordering for every weight and the agreement between rank and min–max normalizations.

## Interpretation Gate

- `construct_index_supported`: balanced panel passes; 50/50 satisfies all four category diagnostics; all central-range weight rankings correlate at least 0.90 with 50/50; rank versus min–max country ordering correlates at least 0.90.
- `construct_index_with_caution`: the panel is valid but at least one robustness or category diagnostic fails.
- `reconsider_index`: no weight satisfies all category diagnostics and central-range minimum rank agreement is below 0.70.

Failure of a category constraint may mean the expert taxonomy covers broader economic access or nutrition concepts than this deliberately narrow cereal-capacity index. It does not authorize changing the data or weights after seeing results.

## Expected Outputs

Default generated directory: `results/generated/capacity/<run-date>`

- `capacity_v2_panel.csv`: 420 country-year rows with raw components, normalization columns, and prespecified scores.
- `weight_sweep_country_ranks.csv`: 101 weights × 30 country-median scores and ranks.
- `weight_sweep_summary.csv`: constraint margins, feasibility, and rank agreement for each weight.
- `country_stability_summary.csv`: temporal and weight sensitivity diagnostics by country.
- `component_rank_correlation.csv`: annual and country-median component rank correlations.
- `summary.json`: scope, diagnostics, verdict, and validation checks.
- `manifest.json`: script/input/output SHA-256 hashes.

This construction step performs no GLM, tree model, SHAP, clustering, outcome
regression, or point imputation. Those analyses, where applicable, are separate
and consume the frozen index output.
