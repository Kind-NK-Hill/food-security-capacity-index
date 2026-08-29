#!/usr/bin/env python3
"""Download the prespecified WDI explanatory panel for the revised analysis.

The script preserves the raw World Bank API responses, writes an explicit
coverage audit for all 31 requested countries, and joins the 30-country
2010-2021 model scope to the already-audited Capacity Score v2 target.  It
does not impute values or fit any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from download_world_bank import COUNTRIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORLD_BANK_API = "https://api.worldbank.org/v2"
USER_AGENT = "FoodSecurityRevised/1.0 (reproducible academic data audit)"
START_YEAR = 2010
END_YEAR = 2021
EXPECTED_YEARS = list(range(START_YEAR, END_YEAR + 1))
MODEL_COUNTRIES = [iso3 for iso3, _ in COUNTRIES if iso3 != "JPN"]

FEATURES: dict[str, dict[str, str]] = {
    "EG.ELC.ACCS.ZS": {
        "feature_name": "electricity_access_pct",
        "concept": "Infrastructure",
        "expected_unit": "% of population",
        "reason_retained": "Development and infrastructure context; not an index component",
    },
    "SP.RUR.TOTL.ZS": {
        "feature_name": "rural_population_pct",
        "concept": "Rural structure",
        "expected_unit": "% of total population",
        "reason_retained": "National settlement structure",
    },
    "SP.RUR.TOTL.ZG": {
        "feature_name": "rural_population_growth_pct",
        "concept": "Rural change",
        "expected_unit": "annual %",
        "reason_retained": "Dynamic demographic structure",
    },
    "SL.AGR.EMPL.FE.ZS": {
        "feature_name": "female_agricultural_employment_pct",
        "concept": "Agricultural labour",
        "expected_unit": "% of female employment",
        "reason_retained": "Labour structure highlighted in the original analysis",
    },
    "AG.LND.ARBL.HA.PC": {
        "feature_name": "arable_land_hectares_per_person",
        "concept": "Land endowment",
        "expected_unit": "hectares per person",
        "reason_retained": "Resource pressure without directly reusing cereal output",
    },
    "AG.CON.FERT.ZS": {
        "feature_name": "fertilizer_consumption_kg_per_hectare_arable_land",
        "concept": "Input intensity",
        "expected_unit": "kilograms per hectare of arable land",
        "reason_retained": "Agricultural input intensity",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and audit the fixed six-feature WDI explanatory panel."
    )
    parser.add_argument(
        "--snapshot-date",
        default=date.today().isoformat(),
        help="Immutable snapshot folder label in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--capacity-input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "capacity" / "capacity_v2_panel.csv",
        help="Capacity panel whose 2010-2021 target columns are joined to the WDI features.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw snapshot directory. Defaults to data/raw/explanatory_wdi/<snapshot-date>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Processed output directory. Defaults to data/processed/explanatory_refresh/<snapshot-date>.",
    )
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    content = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    write_bytes(path, content)


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


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        return response.read()


def data_url(indicator: str) -> str:
    countries = ";".join(iso3 for iso3, _ in COUNTRIES)
    query = urllib.parse.urlencode(
        {
            "date": f"{START_YEAR}:{END_YEAR}",
            "format": "json",
            "per_page": "20000",
        }
    )
    return f"{WORLD_BANK_API}/country/{countries}/indicator/{indicator}?{query}"


def metadata_url(indicator: str) -> str:
    return f"{WORLD_BANK_API}/indicator/{indicator}?format=json"


def parse_data_payload(content: bytes, url: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected World Bank data response for {url}")
    page = payload[0]
    records = payload[1] or []
    if int(page.get("page", 0)) != 1 or int(page.get("pages", 0)) != 1:
        raise RuntimeError(f"Unexpected pagination for {url}: {page}")
    return page, records


def parse_metadata_payload(content: bytes, url: str) -> dict[str, Any]:
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise RuntimeError(f"Unexpected World Bank metadata response for {url}")
    return payload[1][0]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def raw_file_record(path: Path, url: str, content: bytes) -> dict[str, Any]:
    return {
        "path": portable_path(path),
        "url": url,
        "sha256": sha256_bytes(content),
        "bytes": len(content),
    }


def main() -> int:
    args = parse_args()
    datetime.strptime(args.snapshot_date, "%Y-%m-%d")

    raw_root = (
        args.raw_dir.resolve()
        if args.raw_dir is not None
        else PROJECT_ROOT / "data" / "raw" / "explanatory_wdi" / args.snapshot_date
    )
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "data" / "processed" / "explanatory_refresh" / args.snapshot_date
    )
    if raw_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable raw snapshot: {raw_root}. "
            "Use a new snapshot date after inspecting the existing files."
        )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite processed explanatory panel: {output_root}"
        )

    accessed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    country_names = dict(COUNTRIES)
    expected_keys = {
        (iso3, year) for iso3, _ in COUNTRIES for year in EXPECTED_YEARS
    }
    normalized_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    lastupdated_values: set[str] = set()

    for code, config in FEATURES.items():
        indicator_data_url = data_url(code)
        indicator_metadata_url = metadata_url(code)
        data_content = fetch(indicator_data_url)
        metadata_content = fetch(indicator_metadata_url)

        data_path = raw_root / "indicator" / f"{code}.data.json"
        metadata_path = raw_root / "metadata" / f"{code}.metadata.json"
        write_bytes(data_path, data_content)
        write_bytes(metadata_path, metadata_content)
        raw_files.extend(
            [
                raw_file_record(data_path, indicator_data_url, data_content),
                raw_file_record(metadata_path, indicator_metadata_url, metadata_content),
            ]
        )

        page, records = parse_data_payload(data_content, indicator_data_url)
        metadata = parse_metadata_payload(metadata_content, indicator_metadata_url)
        lastupdated = str(page.get("lastupdated", ""))
        lastupdated_values.add(lastupdated)

        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for record in records:
            iso3 = str(record.get("countryiso3code", ""))
            year_text = record.get("date")
            if iso3 not in country_names or year_text is None:
                continue
            key = (iso3, int(year_text))
            if key in by_key:
                raise ValueError(f"Duplicate WDI record for {code}, {key}")
            by_key[key] = record
        unexpected_keys = set(by_key) - expected_keys
        if unexpected_keys:
            raise ValueError(f"Unexpected country-year keys for {code}: {unexpected_keys}")

        for iso3, requested_country in COUNTRIES:
            for year in EXPECTED_YEARS:
                record = by_key.get((iso3, year), {})
                value = record.get("value")
                normalized_rows.append(
                    {
                        "indicator_code": code,
                        "feature_name": config["feature_name"],
                        "concept": config["concept"],
                        "indicator_name": metadata.get("name", ""),
                        "iso3": iso3,
                        "country": requested_country,
                        "api_country_name": (record.get("country") or {}).get("value", ""),
                        "year": year,
                        "value": None if value is None else float(value),
                        "is_missing": value is None,
                        "obs_status": record.get("obs_status", ""),
                        "decimal": record.get("decimal", ""),
                        "unit": metadata.get("unit", ""),
                        "wdi_lastupdated": lastupdated,
                        "api_url": indicator_data_url,
                        "metadata_url": indicator_metadata_url,
                        "accessed_at_utc": accessed_at,
                    }
                )

        source = metadata.get("source") or {}
        dictionary_rows.append(
            {
                "indicator_code": code,
                "feature_name": config["feature_name"],
                "concept": config["concept"],
                "official_indicator_name": metadata.get("name", ""),
                "official_unit": metadata.get("unit", ""),
                "expected_unit_description": config["expected_unit"],
                "source_note": metadata.get("sourceNote", ""),
                "source_organization": metadata.get("sourceOrganization", ""),
                "source_id": source.get("id", ""),
                "source_name": source.get("value", ""),
                "reason_retained": config["reason_retained"],
                "analysis_role": "prespecified explanatory association feature",
                "wdi_lastupdated": lastupdated,
                "api_url": indicator_data_url,
                "metadata_url": indicator_metadata_url,
                "accessed_at_utc": accessed_at,
            }
        )

    long = pd.DataFrame(normalized_rows).sort_values(
        ["indicator_code", "iso3", "year"], ignore_index=True
    )
    expected_rows = len(FEATURES) * len(COUNTRIES) * len(EXPECTED_YEARS)
    if len(long) != expected_rows:
        raise ValueError(f"Expected {expected_rows} normalized rows, found {len(long)}")
    if long.duplicated(["indicator_code", "iso3", "year"]).any():
        raise ValueError("Normalized WDI panel contains duplicate feature-country-year keys")

    coverage_records: list[dict[str, Any]] = []
    for (code, feature_name, iso3, country), group in long.groupby(
        ["indicator_code", "feature_name", "iso3", "country"], sort=True
    ):
        observed_years = sorted(group.loc[group["value"].notna(), "year"].astype(int))
        missing_years = sorted(set(EXPECTED_YEARS) - set(observed_years))
        coverage_records.append(
            {
                "indicator_code": code,
                "feature_name": feature_name,
                "iso3": iso3,
                "country": country,
                "included_in_30_country_model_scope": iso3 in MODEL_COUNTRIES,
                "expected_years": len(EXPECTED_YEARS),
                "observed_years": len(observed_years),
                "missing_year_count": len(missing_years),
                "missing_years": ";".join(map(str, missing_years)),
                "coverage_fraction": len(observed_years) / len(EXPECTED_YEARS),
                "minimum_required_observed_years": 8,
                "country_gate_pass": len(observed_years) >= 8,
            }
        )
    coverage = pd.DataFrame(coverage_records).sort_values(
        ["indicator_code", "iso3"], ignore_index=True
    )

    coverage_summary_records: list[dict[str, Any]] = []
    for code, config in FEATURES.items():
        model_rows = long.loc[
            long["indicator_code"].eq(code) & long["iso3"].isin(MODEL_COUNTRIES)
        ]
        country_rows = coverage.loc[
            coverage["indicator_code"].eq(code)
            & coverage["included_in_30_country_model_scope"]
        ]
        observed = int(model_rows["value"].notna().sum())
        total = int(len(model_rows))
        overall_coverage = observed / total
        minimum_country_years = int(country_rows["observed_years"].min())
        gate_pass = overall_coverage >= 0.80 and minimum_country_years >= 8
        coverage_summary_records.append(
            {
                "indicator_code": code,
                "feature_name": config["feature_name"],
                "model_scope_countries": len(MODEL_COUNTRIES),
                "model_scope_expected_cells": total,
                "model_scope_observed_cells": observed,
                "model_scope_missing_cells": total - observed,
                "model_scope_coverage_fraction": overall_coverage,
                "minimum_country_observed_years": minimum_country_years,
                "overall_coverage_threshold": 0.80,
                "country_year_threshold": 8,
                "feature_gate_pass": gate_pass,
                "gate_reason": (
                    "pass"
                    if gate_pass
                    else "removed before modeling: overall coverage <80% or a country has <8 observed years"
                ),
            }
        )
    coverage_summary = pd.DataFrame(coverage_summary_records)

    missing_cells = long.loc[long["value"].isna()].copy()
    missing_cells = missing_cells[
        [
            "indicator_code",
            "feature_name",
            "iso3",
            "country",
            "year",
            "wdi_lastupdated",
            "api_url",
            "accessed_at_utc",
        ]
    ]

    target_path = args.capacity_input.resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Missing capacity input: {target_path}")
    target = pd.read_csv(target_path)
    target = target.loc[target["year"].between(START_YEAR, END_YEAR)].copy()
    target_columns = [
        "iso3",
        "country",
        "year",
        "legacy_category",
        "capacity_v2_rank_50_50",
        "capacity_v2_rank_legacy_56_44",
        "capacity_v2_minmax_50_50",
        "ssr_percentile_rank",
        "production_per_person_percentile_rank",
    ]
    target = target[target_columns].sort_values(["iso3", "year"], ignore_index=True)
    if len(target) != len(MODEL_COUNTRIES) * len(EXPECTED_YEARS):
        raise ValueError(f"Unexpected Capacity v2 model-window rows: {len(target)}")
    if set(target["iso3"].unique()) != set(MODEL_COUNTRIES):
        raise ValueError("Capacity v2 country scope does not match the prespecified 30 countries")
    if target[["iso3", "year"]].duplicated().any() or target.isna().any().any():
        raise ValueError("Capacity v2 model-window target keys are duplicate or incomplete")

    model_long = long.loc[long["iso3"].isin(MODEL_COUNTRIES)].copy()
    feature_wide = model_long.pivot(
        index=["iso3", "country", "year"], columns="feature_name", values="value"
    ).reset_index()
    feature_wide.columns.name = None
    panel = target.merge(
        feature_wide.drop(columns="country"),
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    ).sort_values(["iso3", "year"], ignore_index=True)
    if len(panel) != 360 or panel[["iso3", "year"]].duplicated().any():
        raise ValueError("Joined explanatory panel is not a unique 30-country x 12-year grid")

    output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "explanatory_panel": output_root / "explanatory_panel.csv",
        "explanatory_long": output_root / "explanatory_long.csv",
        "feature_dictionary": output_root / "feature_dictionary.csv",
        "coverage": output_root / "coverage.csv",
        "coverage_summary": output_root / "coverage_summary.csv",
        "missing_cells": output_root / "missing_cells.csv",
    }
    write_csv(panel, paths["explanatory_panel"])
    write_csv(long, paths["explanatory_long"])
    write_csv(pd.DataFrame(dictionary_rows), paths["feature_dictionary"])
    write_csv(coverage, paths["coverage"])
    write_csv(coverage_summary, paths["coverage_summary"])
    write_csv(missing_cells, paths["missing_cells"])

    passed_features = coverage_summary.loc[
        coverage_summary["feature_gate_pass"], "feature_name"
    ].tolist()
    failed_features = coverage_summary.loc[
        ~coverage_summary["feature_gate_pass"], "feature_name"
    ].tolist()
    summary = {
        "artifact": "prespecified official WDI explanatory panel for revised Food Security analysis",
        "scope": {
            "api_download_countries": len(COUNTRIES),
            "api_download_years": f"{START_YEAR}-{END_YEAR}",
            "api_download_features": len(FEATURES),
            "normalized_rows": len(long),
            "model_scope_countries": len(MODEL_COUNTRIES),
            "model_scope_exclusion": {
                "iso3": "JPN",
                "reason": "Capacity v2 lacks official FAO cereal SSR for Japan",
            },
            "model_panel_rows": len(panel),
        },
        "semantics": {
            "source": "World Bank Indicators API v2",
            "values": "published API values; no manual replacement, interpolation, edge fill, or outcome-derived component",
            "missingness": "null API cells are preserved and listed explicitly",
            "later_fold_preprocessing": "for a retained incomplete feature only, median imputation and a missingness indicator must be fitted within each training fold",
            "target": "Capacity Score v2 annual percentile-rank 50/50 composite",
            "analysis_claim": "explanatory association, not forecasting or causality",
        },
        "data_gate": {
            "overall_coverage_threshold": 0.80,
            "minimum_observed_years_per_country": 8,
            "passed_features": passed_features,
            "failed_features": failed_features,
            "all_six_features_pass": len(passed_features) == len(FEATURES),
            "modeling_allowed": len(passed_features) >= 3,
            "coverage": coverage_summary.to_dict(orient="records"),
        },
        "provenance": {
            "snapshot_date": args.snapshot_date,
            "accessed_at_utc": accessed_at,
            "wdi_lastupdated_values": sorted(lastupdated_values),
            "raw_files": raw_files,
            "capacity_v2_input": {
                "path": portable_path(target_path),
                "sha256": sha256_file(target_path),
                "bytes": target_path.stat().st_size,
            },
        },
        "validation": {
            "expected_normalized_rows": expected_rows,
            "normalized_rows": len(long),
            "unique_feature_country_year_keys": int(
                long[["indicator_code", "iso3", "year"]].drop_duplicates().shape[0]
            ),
            "raw_missing_cells_all_31_countries": int(long["value"].isna().sum()),
            "raw_missing_cells_model_30_countries": int(
                model_long["value"].isna().sum()
            ),
            "model_panel_rows": len(panel),
            "target_missing_cells": int(panel["capacity_v2_rank_50_50"].isna().sum()),
            "point_imputation_performed": False,
            "model_fitted": False,
        },
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)

    manifest = {
        "script": {
            "path": portable_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
        },
        "raw_files": raw_files,
        "inputs": [summary["provenance"]["capacity_v2_input"]],
        "outputs": [
            {
                "path": portable_path(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in [*paths.values(), summary_path]
        ],
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "raw_snapshot": portable_path(raw_root),
                "output_dir": portable_path(output_root),
                "wdi_lastupdated_values": sorted(lastupdated_values),
                "raw_missing_cells_all_31": int(long["value"].isna().sum()),
                "raw_missing_cells_model_30": int(model_long["value"].isna().sum()),
                "features_passed": passed_features,
                "features_failed": failed_features,
                "modeling_allowed": summary["data_gate"]["modeling_allowed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
