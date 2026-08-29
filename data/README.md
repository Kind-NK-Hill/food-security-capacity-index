# Data layout

Only compact processed inputs required by `scripts/run_analysis.py` are tracked.

```text
processed/
├── capacity/
│   ├── capacity_v2_panel.csv
│   ├── country_stability_summary.csv
│   └── weight_sweep_summary.csv
├── explanatory/
│   ├── explanatory_panel.csv
│   ├── feature_dictionary.csv
│   ├── coverage.csv
│   ├── summary.json
│   └── manifest.json
└── outcomes/
    └── outcome_panel.csv
```

The outcome panel contains only PoU/FIES fields used by the main analysis.
GHI row-level values are not redistributed. Raw downloads and refresh outputs
are ignored by Git; use the acquisition scripts when a new source vintage is
required.

Key semantics:

- `capacity_v2_panel.csv`: 30 countries x 14 years (2010-2023), 420 rows.
- `explanatory_panel.csv`: 30 countries x 12 years (2010-2021), 360 rows,
  complete on six prespecified WDI predictors.
- `outcome_panel.csv`: 420 ISO3-year rows; PoU censoring and FIES publication
  status remain explicit.
- Japan is retained in acquisition scope but excluded from modern v2 because
  its FAOSTAT Food Balances cereal series is absent for 2010-2023.

See `../DATA_SOURCES.md` for licenses and attribution.
