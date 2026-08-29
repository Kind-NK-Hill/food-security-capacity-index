#!/usr/bin/env python3
"""Download and normalize official FAOSTAT outcomes.

The script preserves interval censoring, suppression, and absence as distinct
states. It never converts '<2.5' to a point estimate and never imputes
suppressed or absent values. GHI is optional and disabled by default because
its redistribution terms differ from the FAOSTAT statistical-data license.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COUNTRIES = [
    ("ARG", "Argentina", "032"),
    ("BGD", "Bangladesh", "050"),
    ("BRA", "Brazil", "076"),
    ("CHN", "China", "156"),
    ("COL", "Colombia", "170"),
    ("COD", "Congo, Dem. Rep.", "180"),
    ("EGY", "Egypt, Arab Rep.", "818"),
    ("ETH", "Ethiopia", "231"),
    ("FRA", "France", "250"),
    ("DEU", "Germany", "276"),
    ("IND", "India", "356"),
    ("IDN", "Indonesia", "360"),
    ("IRN", "Iran, Islamic Rep.", "364"),
    ("ITA", "Italy", "380"),
    ("JPN", "Japan", "392"),
    ("KEN", "Kenya", "404"),
    ("KOR", "Korea, Rep.", "410"),
    ("MYS", "Malaysia", "458"),
    ("MEX", "Mexico", "484"),
    ("MMR", "Myanmar", "104"),
    ("NGA", "Nigeria", "566"),
    ("PAK", "Pakistan", "586"),
    ("PHL", "Philippines", "608"),
    ("RUS", "Russian Federation", "643"),
    ("ZAF", "South Africa", "710"),
    ("TZA", "Tanzania", "834"),
    ("THA", "Thailand", "764"),
    ("TUR", "Turkiye", "792"),
    ("GBR", "United Kingdom", "826"),
    ("USA", "United States", "840"),
    ("VNM", "Viet Nam", "704"),
]

FAOSTAT_FS_CATALOG = "https://data.fao.org/catalog/dataset/955d6564-40a9-48b4-b51b-f19d65bb3539"
FAOSTAT_FS_QUERY = (
    "https://data.apps.fao.org/catalog/dataset/ca2d4c71-d1e8-46b8-9a4f-588a0604e195/"
    "resource/efe51d5d-51e1-4293-9042-75a5826af7c8/download/food-security-indicators-fs-query.sql"
)
FAOSTAT_FS_SCHEMA = (
    "https://data.apps.fao.org/catalog/dataset/ca2d4c71-d1e8-46b8-9a4f-588a0604e195/"
    "resource/996c642a-8148-4829-ba6c-bdbf9808647b/download/food-security-fs-terriajs-schema.json"
)
FAOSTAT_FS_METADATA = "https://files-faostat.fao.org/production/FS/Descriptions_and_Metadata.xlsx"

FAOSTAT_FBS_CATALOG = "https://data.fao.org/catalog/iso/2f264bb6-1238-459a-bf8b-0e2d0a16804a"
FAOSTAT_FBS_QUERY = (
    "https://data.apps.fao.org/catalog/dataset/5c00a4e6-0ec8-4191-a0c0-a7cd5fda3674/"
    "resource/91b9d43c-55c4-4a2a-9b25-78f2fabba28b/download/fct-fbs-food-balances.query.sql"
)
GHI_2025_URL = "https://www.globalhungerindex.org/xlsx/2025.xlsx"
GHI_METHODOLOGY_URL = "https://www.globalhungerindex.org/methodology.html"

FS_ITEMS = {
    "210041": {
        "slug": "pou_3year",
        "name": "Prevalence of undernourishment (percent) (3-year average)",
        "years": range(2001, 2025),
    },
    "210091": {
        "slug": "fies_moderate_or_severe_3year",
        "name": "Prevalence of moderate or severe food insecurity in the total population (percent) (3-year average)",
        "years": range(2015, 2025),
    },
    "21035": {
        "slug": "cereal_import_dependency_3year",
        "name": "Cereal import dependency ratio (percent) (3-year average)",
        "years": range(2001, 2023),
    },
}

GHI_NAME_MAPPING = {
    "Türkiye": "Turkiye",
    "Iran (Islamic Republic of)": "Iran, Islamic Rep.",
    "Tanzania (United Rep. of)": "Tanzania",
    "Dem. Rep. of the Congo": "Congo, Dem. Rep.",
    "Egypt": "Egypt, Arab Rep.",
}

USER_AGENT = "CerealCapacityIndex/1.0 (reproducible academic data audit)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-date",
        default=date.today().isoformat(),
        help="Version folder in YYYY-MM-DD form (default: today's date).",
    )
    parser.add_argument(
        "--include-ghi",
        action="store_true",
        help="Also download the third-party GHI workbook for local-only sensitivity analysis.",
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


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )


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


def fetch(url: str, attempts: int = 4) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                headers = {key.lower(): value for key, value in response.headers.items()}
                return response.read(), headers
        except Exception as error:  # network retry is intentionally bounded
            last_error = error
            if attempt == attempts - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Unable to fetch {url}: {last_error}")


def bigquery_url(item_code: str, query_url: str) -> str:
    query = urllib.parse.urlencode(
        {"download": "true", "item_code": item_code, "sql_url": query_url}
    )
    return f"https://api.data.apps.fao.org/api/v2/bigquery?{query}"


def country_frame() -> pd.DataFrame:
    return pd.DataFrame(COUNTRIES, columns=["iso3", "country", "m49_code"])


def parse_api_csv(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)


def parse_value(raw_value: object, row_exists: bool, flag: str = "") -> dict[str, Any]:
    raw = "" if raw_value is None else str(raw_value).strip()
    flag_lower = flag.lower()
    if not row_exists:
        return {
            "value_status": "absent_from_api_response",
            "exact_value": np.nan,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
        }
    if not raw:
        status = "suppressed" if "suppress" in flag_lower else "published_missing"
        return {
            "value_status": status,
            "exact_value": np.nan,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
        }
    censored = re.fullmatch(r"([<>])\s*([0-9]+(?:\.[0-9]+)?)", raw)
    if censored:
        operator, threshold_text = censored.groups()
        threshold = float(threshold_text)
        if operator == "<":
            return {
                "value_status": "left_censored",
                "exact_value": np.nan,
                "lower_bound": 0.0,
                "upper_bound": threshold,
            }
        return {
            "value_status": "right_censored",
            "exact_value": np.nan,
            "lower_bound": threshold,
            "upper_bound": np.nan,
        }
    value = float(raw.replace(",", ""))
    return {
        "value_status": "observed_exact",
        "exact_value": value,
        "lower_bound": value,
        "upper_bound": value,
    }


def normalize_fs_item(
    raw: pd.DataFrame,
    item_code: str,
    source_url: str,
    accessed_at: str,
) -> pd.DataFrame:
    config = FS_ITEMS[item_code]
    countries = country_frame()
    raw = raw.copy()
    raw["m49_code"] = raw["m49_code"].astype(str).str.zfill(3)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    target = raw.loc[raw["m49_code"].isin(countries["m49_code"])].copy()
    if target.duplicated(["m49_code", "year"]).any():
        raise ValueError(f"Duplicate FAOSTAT keys for item {item_code}")
    lookup = target.set_index(["m49_code", "year"]).to_dict(orient="index")

    records: list[dict[str, Any]] = []
    for iso3, country, m49 in COUNTRIES:
        for year in config["years"]:
            row = lookup.get((m49, year))
            exists = row is not None
            flag = "" if row is None else str(row.get("value__flag", ""))
            raw_value = "" if row is None else row.get("value_", "")
            parsed = parse_value(raw_value, exists, flag)
            records.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "m49_code": m49,
                    "faostat_area_code": "" if row is None else row.get("faostat", ""),
                    "item_code": item_code,
                    "item": config["name"],
                    "year_period": "" if row is None else row.get("year_period", ""),
                    "year": year,
                    "raw_value": raw_value,
                    **parsed,
                    "value_flag": flag,
                    "unit": "percent",
                    "source_url": source_url,
                    "accessed_at_utc": accessed_at,
                }
            )
    result = pd.DataFrame(records).sort_values(["iso3", "year"], ignore_index=True)
    expected = 31 * len(list(config["years"]))
    if len(result) != expected or result.duplicated(["iso3", "year"]).any():
        raise ValueError(f"Unexpected normalized grid for item {item_code}")
    if item_code == "21035":
        result["cereal_self_sufficiency_candidate_pct"] = 100.0 - result["exact_value"]
        result["derived_semantics"] = (
            "100 - official cereal import dependency ratio; may exceed 100 for net exporters"
        )
    return result


def normalize_fbs(
    raw: pd.DataFrame,
    source_url: str,
    accessed_at: str,
) -> pd.DataFrame:
    countries = country_frame()
    raw = raw.copy()
    raw["m49_code"] = raw["m49_code"].astype(str).str.zfill(3)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    target = raw.loc[raw["m49_code"].isin(countries["m49_code"])].copy()
    if target.duplicated(["m49_code", "year"]).any():
        raise ValueError("Duplicate FAOSTAT Food Balances keys")
    lookup = target.set_index(["m49_code", "year"]).to_dict(orient="index")

    records: list[dict[str, Any]] = []
    for iso3, country, m49 in COUNTRIES:
        for year in range(2010, 2024):
            row = lookup.get((m49, year))
            exists = row is not None
            production = np.nan if row is None else pd.to_numeric(
                row.get("production_1000_tonnes", ""), errors="coerce"
            )
            imports = np.nan if row is None else pd.to_numeric(
                row.get("import_quantity_1000_tonnes", ""), errors="coerce"
            )
            exports = np.nan if row is None else pd.to_numeric(
                row.get("export_quantity_1000_tonnes", ""), errors="coerce"
            )
            complete = all(pd.notna(value) for value in [production, imports, exports])
            denominator = production + imports - exports if complete else np.nan
            ssr = production / denominator * 100.0 if complete and denominator > 0 else np.nan
            if not exists:
                status = "absent_from_api_response"
            elif not complete:
                status = "incomplete_components"
            elif not denominator > 0:
                status = "invalid_nonpositive_denominator"
            else:
                status = "observed_components"
            records.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "m49_code": m49,
                    "faostat_area_code": "" if row is None else row.get("faostat", ""),
                    "item_code": "2905",
                    "item": "Cereals - Excluding Beer",
                    "year": year,
                    "production_1000_tonnes": production,
                    "production_flag": "" if row is None else row.get("production_1000_tonnes_flag", ""),
                    "import_1000_tonnes": imports,
                    "import_flag": "" if row is None else row.get("import_quantity_1000_tonnes_flag", ""),
                    "export_1000_tonnes": exports,
                    "export_flag": "" if row is None else row.get("export_quantity_1000_tonnes_flag", ""),
                    "fbs_cereal_ssr_candidate_pct": ssr,
                    "value_status": status,
                    "formula": "production / (production + imports - exports) * 100",
                    "source_url": source_url,
                    "accessed_at_utc": accessed_at,
                }
            )
    result = pd.DataFrame(records).sort_values(["iso3", "year"], ignore_index=True)
    if len(result) != 31 * 14 or result.duplicated(["iso3", "year"]).any():
        raise ValueError("Unexpected normalized Food Balances grid")
    return result


def normalize_ghi(content: bytes, source_url: str, accessed_at: str) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(content), sheet_name=0, header=2)
    raw.columns = [str(column) for column in raw.columns]
    required = {"Country", "2000", "2008", "2016", "2025"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Unexpected GHI columns: {raw.columns.tolist()}")
    raw = raw[["Country", "2000", "2008", "2016", "2025"]].copy()
    raw["Country"] = raw["Country"].replace(GHI_NAME_MAPPING)
    targets = {country for _, country, _ in COUNTRIES}
    raw = raw.loc[raw["Country"].isin(targets)]
    if raw["Country"].duplicated().any():
        raise ValueError("Duplicate target country in GHI workbook")
    lookup = raw.set_index("Country").to_dict(orient="index")

    records: list[dict[str, Any]] = []
    for iso3, country, m49 in COUNTRIES:
        country_row = lookup.get(country)
        for year in [2000, 2008, 2016, 2025]:
            exists = country_row is not None
            raw_value = "" if country_row is None else country_row.get(str(year), "")
            parsed = parse_value(raw_value, exists)
            records.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "m49_code": m49,
                    "year": year,
                    "raw_value": raw_value,
                    **parsed,
                    "unit": "GHI points on 0-100 scale",
                    "source_url": source_url,
                    "methodology_url": GHI_METHODOLOGY_URL,
                    "accessed_at_utc": accessed_at,
                }
            )
    result = pd.DataFrame(records).sort_values(["iso3", "year"], ignore_index=True)
    if len(result) != 31 * 4 or result.duplicated(["iso3", "year"]).any():
        raise ValueError("Unexpected GHI 31-country reference grid")
    return result


def empty_ghi(accessed_at: str) -> pd.DataFrame:
    """Return an explicit all-absent GHI grid when the optional source is disabled."""
    records: list[dict[str, Any]] = []
    for iso3, country, m49 in COUNTRIES:
        for year in [2000, 2008, 2016, 2025]:
            records.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "m49_code": m49,
                    "year": year,
                    "raw_value": "",
                    **parse_value("", row_exists=False),
                    "unit": "GHI points on 0-100 scale",
                    "source_url": "",
                    "methodology_url": GHI_METHODOLOGY_URL,
                    "accessed_at_utc": accessed_at,
                }
            )
    return pd.DataFrame(records).sort_values(["iso3", "year"], ignore_index=True)


def add_source_columns(panel: pd.DataFrame, source: pd.DataFrame, prefix: str, columns: list[str]) -> pd.DataFrame:
    selected = source[["iso3", "year", *columns]].rename(
        columns={column: f"{prefix}_{column}" for column in columns}
    )
    return panel.merge(selected, on=["iso3", "year"], how="left", validate="one_to_one")


def build_panel(
    pou: pd.DataFrame,
    fies: pd.DataFrame,
    cidr: pd.DataFrame,
    fbs: pd.DataFrame,
    ghi: pd.DataFrame,
) -> pd.DataFrame:
    records = [
        {"iso3": iso3, "country": country, "m49_code": m49, "year": year}
        for iso3, country, m49 in COUNTRIES
        for year in range(2000, 2026)
    ]
    panel = pd.DataFrame(records)
    panel = add_source_columns(
        panel,
        pou,
        "pou",
        ["year_period", "raw_value", "exact_value", "lower_bound", "upper_bound", "value_status", "value_flag"],
    )
    panel = add_source_columns(
        panel,
        fies,
        "fies",
        ["year_period", "raw_value", "exact_value", "lower_bound", "upper_bound", "value_status", "value_flag"],
    )
    panel = add_source_columns(
        panel,
        cidr,
        "cidr",
        ["year_period", "raw_value", "exact_value", "value_status", "value_flag", "cereal_self_sufficiency_candidate_pct"],
    )
    panel = add_source_columns(
        panel,
        fbs,
        "fbs",
        [
            "production_1000_tonnes",
            "import_1000_tonnes",
            "export_1000_tonnes",
            "fbs_cereal_ssr_candidate_pct",
            "value_status",
        ],
    )
    panel = add_source_columns(
        panel,
        ghi,
        "ghi",
        ["raw_value", "exact_value", "lower_bound", "upper_bound", "value_status"],
    )
    return panel.sort_values(["iso3", "year"], ignore_index=True)


def build_country_coverage(
    pou: pd.DataFrame,
    fies: pd.DataFrame,
    cidr: pd.DataFrame,
    fbs: pd.DataFrame,
    ghi: pd.DataFrame,
) -> pd.DataFrame:
    base = country_frame()

    def counts(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return (
            frame.groupby("iso3")
            .agg(
                **{
                    f"{prefix}_published": (
                        "value_status",
                        lambda values: int((~values.eq("absent_from_api_response")).sum()),
                    ),
                    f"{prefix}_exact": ("exact_value", lambda values: int(values.notna().sum())),
                    f"{prefix}_censored": (
                        "value_status",
                        lambda values: int(values.isin(["left_censored", "right_censored"]).sum()),
                    ),
                    f"{prefix}_suppressed": (
                        "value_status",
                        lambda values: int(values.eq("suppressed").sum()),
                    ),
                }
            )
            .reset_index()
        )

    for frame, prefix in [(pou, "pou"), (fies, "fies"), (cidr, "cidr"), (ghi, "ghi")]:
        base = base.merge(counts(frame, prefix), on="iso3", how="left", validate="one_to_one")
    fbs_counts = (
        fbs.groupby("iso3")["fbs_cereal_ssr_candidate_pct"]
        .count()
        .rename("fbs_ssr_exact")
        .reset_index()
    )
    base = base.merge(fbs_counts, on="iso3", how="left", validate="one_to_one")
    base["pou_primary_outcome_published_complete"] = base["pou_published"].eq(24)
    base["fies_secondary_has_any_exact"] = base["fies_exact"].gt(0)
    base["cidr_has_complete_22_periods"] = base["cidr_exact"].eq(22)
    base["fbs_ssr_has_complete_2010_2023"] = base["fbs_ssr_exact"].eq(14)
    base["ghi_has_any_published_reference"] = base["ghi_published"].gt(0)
    return base.sort_values("iso3", ignore_index=True)


def coverage_row(
    name: str,
    role: str,
    item_code: str,
    frame: pd.DataFrame,
    expected_years: int,
    semantics: str,
    decision: str,
) -> dict[str, Any]:
    status = frame["value_status"]
    country_counts = frame.groupby("iso3").agg(
        published=("value_status", lambda values: int((~values.eq("absent_from_api_response")).sum())),
        exact=("exact_value", lambda values: int(values.notna().sum())),
    )
    exceptions = [
        f"{iso}:published={int(row.published)},exact={int(row.exact)}"
        for iso, row in country_counts.iterrows()
        if int(row.published) != expected_years or int(row.exact) != expected_years
    ]
    return {
        "source_variable": name,
        "role": role,
        "official_item_code": item_code,
        "grid_rows": int(len(frame)),
        "observed_exact_rows": int(status.eq("observed_exact").sum()),
        "left_censored_rows": int(status.eq("left_censored").sum()),
        "suppressed_rows": int(status.eq("suppressed").sum()),
        "absent_rows": int(status.eq("absent_from_api_response").sum()),
        "other_missing_rows": int(
            status.isin(["published_missing", "incomplete_components", "invalid_nonpositive_denominator"]).sum()
        ),
        "countries_any_published": int((country_counts["published"] > 0).sum()),
        "countries_any_exact": int((country_counts["exact"] > 0).sum()),
        "countries_complete_published": int((country_counts["published"] == expected_years).sum()),
        "countries_complete_exact": int((country_counts["exact"] == expected_years).sum()),
        "country_exceptions": "; ".join(exceptions),
        "time_and_value_semantics": semantics,
        "decision": decision,
    }


def fbs_coverage_row(frame: pd.DataFrame) -> dict[str, Any]:
    by_country = frame.groupby("iso3")["fbs_cereal_ssr_candidate_pct"].count()
    exceptions = [f"{iso}:exact={int(count)}" for iso, count in by_country.items() if count != 14]
    return {
        "source_variable": "FAOSTAT Food Balances cereal SSR candidate",
        "role": "availability sensitivity",
        "official_item_code": "2905",
        "grid_rows": int(len(frame)),
        "observed_exact_rows": int(frame["fbs_cereal_ssr_candidate_pct"].notna().sum()),
        "left_censored_rows": 0,
        "suppressed_rows": 0,
        "absent_rows": int(frame["value_status"].eq("absent_from_api_response").sum()),
        "other_missing_rows": int(
            frame["value_status"].isin(["incomplete_components", "invalid_nonpositive_denominator"]).sum()
        ),
        "countries_any_published": int((by_country > 0).sum()),
        "countries_any_exact": int((by_country > 0).sum()),
        "countries_complete_published": int((by_country == 14).sum()),
        "countries_complete_exact": int((by_country == 14).sum()),
        "country_exceptions": "; ".join(exceptions),
        "time_and_value_semantics": (
            "annual 2010-2023; 1000 tonnes; production/(production+imports-exports)*100; all available aggregate rows estimated"
        ),
        "decision": "retain as 30-country sensitivity; Japan remains absent",
    }


def ghi_coverage_row(frame: pd.DataFrame) -> dict[str, Any]:
    by_country = frame.groupby("iso3").agg(
        published=("value_status", lambda values: int((~values.eq("absent_from_api_response")).sum())),
        exact=("exact_value", lambda values: int(values.notna().sum())),
    )
    exceptions = [
        f"{iso}:published={int(row.published)},exact={int(row.exact)}"
        for iso, row in by_country.iterrows()
        if int(row.published) != 4 or int(row.exact) != 4
    ]
    return {
        "source_variable": "Global Hunger Index 2025 edition",
        "role": "sparse external benchmark",
        "official_item_code": "not applicable",
        "grid_rows": int(len(frame)),
        "observed_exact_rows": int(frame["value_status"].eq("observed_exact").sum()),
        "left_censored_rows": int(frame["value_status"].eq("left_censored").sum()),
        "suppressed_rows": 0,
        "absent_rows": int(frame["value_status"].eq("absent_from_api_response").sum()),
        "other_missing_rows": 0,
        "countries_any_published": int((by_country["published"] > 0).sum()),
        "countries_any_exact": int((by_country["exact"] > 0).sum()),
        "countries_complete_published": int((by_country["published"] == 4).sum()),
        "countries_complete_exact": int((by_country["exact"] == 4).sum()),
        "country_exceptions": "; ".join(exceptions),
        "time_and_value_semantics": (
            "same-edition reference points 2000/2008/2016/2025; '<5' retained as [0,5); no annual interpolation"
        ),
        "decision": "retain as 24-country four-point benchmark; do not synthesize seven absent countries",
    }


def validate_status(frame: pd.DataFrame, name: str) -> None:
    observed = frame["value_status"].eq("observed_exact")
    censored = frame["value_status"].isin(["left_censored", "right_censored"])
    if frame.loc[observed, "exact_value"].isna().any():
        raise ValueError(f"{name}: observed rows without exact values")
    if frame.loc[censored, "exact_value"].notna().any():
        raise ValueError(f"{name}: censored rows incorrectly assigned point values")
    if frame["value_status"].eq("").any() or frame["value_status"].isna().any():
        raise ValueError(f"{name}: unlabeled missing state")


def record_download(
    records: list[dict[str, Any]],
    name: str,
    url: str,
    path: Path,
    content: bytes,
    headers: dict[str, str],
    accessed_at: str,
) -> None:
    records.append(
        {
            "name": name,
            "url": url,
            "path": str(path),
            "accessed_at_utc": accessed_at,
            "content_type": headers.get("content-type", ""),
            "last_modified_header": headers.get("last-modified", ""),
            "etag": headers.get("etag", ""),
            "bytes": len(content),
            "sha256": sha256_bytes(content),
        }
    )


def main() -> int:
    args = parse_args()
    datetime.strptime(args.snapshot_date, "%Y-%m-%d")
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw" / "official_outcomes" / args.snapshot_date
    processed_root = project_root / "data" / "processed" / "official_outcomes" / args.snapshot_date
    if raw_root.exists() or processed_root.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing official-outcome snapshot. "
            "Use a new --snapshot-date or move the existing directories."
        )
    accessed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    downloads: list[dict[str, Any]] = []

    static_sources = {
        "faostat_food_security_schema": (FAOSTAT_FS_SCHEMA, raw_root / "metadata" / "faostat_food_security_schema.json"),
        "faostat_food_security_query": (FAOSTAT_FS_QUERY, raw_root / "metadata" / "faostat_food_security_query.sql"),
        "faostat_food_security_descriptions": (FAOSTAT_FS_METADATA, raw_root / "metadata" / "faostat_food_security_descriptions.xlsx"),
        "faostat_food_balances_query": (FAOSTAT_FBS_QUERY, raw_root / "metadata" / "faostat_food_balances_query.sql"),
    }
    if args.include_ghi:
        static_sources["ghi_2025_workbook"] = (
            GHI_2025_URL,
            raw_root / "ghi" / "2025.xlsx",
        )
    static_content: dict[str, bytes] = {}
    for name, (url, path) in static_sources.items():
        content, headers = fetch(url)
        write_bytes_atomic(path, content)
        record_download(downloads, name, url, path, content, headers, accessed_at)
        static_content[name] = content

    normalized_fs: dict[str, pd.DataFrame] = {}
    for item_code, config in FS_ITEMS.items():
        url = bigquery_url(item_code, FAOSTAT_FS_QUERY)
        content, headers = fetch(url)
        path = raw_root / "faostat_food_security" / f"{item_code}.{config['slug']}.csv"
        write_bytes_atomic(path, content)
        record_download(downloads, f"faostat_fs_{item_code}", url, path, content, headers, accessed_at)
        normalized_fs[item_code] = normalize_fs_item(
            parse_api_csv(content), item_code, url, accessed_at
        )

    fbs_url = bigquery_url("2905", FAOSTAT_FBS_QUERY)
    fbs_content, fbs_headers = fetch(fbs_url)
    fbs_raw_path = raw_root / "faostat_food_balances" / "2905.cereals_excluding_beer.csv"
    write_bytes_atomic(fbs_raw_path, fbs_content)
    record_download(
        downloads,
        "faostat_fbs_2905",
        fbs_url,
        fbs_raw_path,
        fbs_content,
        fbs_headers,
        accessed_at,
    )
    fbs = normalize_fbs(parse_api_csv(fbs_content), fbs_url, accessed_at)
    ghi = (
        normalize_ghi(static_content["ghi_2025_workbook"], GHI_2025_URL, accessed_at)
        if args.include_ghi
        else empty_ghi(accessed_at)
    )

    pou = normalized_fs["210041"]
    fies = normalized_fs["210091"]
    cidr = normalized_fs["21035"]
    for name, frame in [("PoU", pou), ("FIES", fies), ("CIDR", cidr), ("GHI", ghi)]:
        validate_status(frame, name)

    panel = build_panel(pou, fies, cidr, fbs, ghi)
    country_coverage = build_country_coverage(pou, fies, cidr, fbs, ghi)
    if len(panel) != 31 * 26 or panel.duplicated(["iso3", "year"]).any():
        raise ValueError("Combined outcome/benchmark panel is not a unique 31x26 grid")
    if len(country_coverage) != 31 or country_coverage["iso3"].duplicated().any():
        raise ValueError("Country coverage output is not a unique 31-country table")

    coverage = pd.DataFrame(
        [
            coverage_row(
                "FAOSTAT PoU",
                "primary outcome candidate",
                "210041",
                pou,
                24,
                "center years 2001-2024 represent three-year periods 2000-2002 through 2023-2025; '<2.5' retained as [0,2.5)",
                "allow 31-country censor-aware outcome preprocessing; do not replace '<2.5' with 2.5",
            ),
            coverage_row(
                "FAOSTAT FIES moderate or severe",
                "secondary outcome",
                "210091",
                fies,
                10,
                "center years 2015-2024 represent three-year averages; suppressed and absent are not numeric missing-at-random values",
                "allow unbalanced 28-country validation; do not impute China, India, or Turkiye",
            ),
            coverage_row(
                "FAOSTAT cereal import dependency ratio",
                "availability sensitivity",
                "21035",
                cidr,
                22,
                "center years 2001-2022 represent three-year averages; self-sufficiency candidate equals 100-CIDR and may exceed 100",
                "retain as unbalanced sensitivity; Japan ends at 2007-2009 and Congo starts at 2010-2012",
            ),
            fbs_coverage_row(fbs),
            ghi_coverage_row(ghi),
        ]
    )

    outputs = {
        "country_mapping": processed_root / "country_mapping.csv",
        "pou": processed_root / "pou_31countries_2001_2024.csv",
        "fies": processed_root / "fies_31countries_2015_2024.csv",
        "cidr": processed_root / "cereal_import_dependency_31countries_2001_2022.csv",
        "fbs": processed_root / "fbs_cereal_ssr_31countries_2010_2023.csv",
        "ghi": processed_root / "ghi_2025_31countries_reference_points.csv",
        "panel": processed_root / "outcome_benchmark_source_panel_2000_2025.csv",
        "coverage": processed_root / "coverage_semantics.csv",
        "country_coverage": processed_root / "country_outcome_coverage.csv",
    }
    write_csv(country_frame(), outputs["country_mapping"])
    write_csv(pou, outputs["pou"])
    write_csv(fies, outputs["fies"])
    write_csv(cidr, outputs["cidr"])
    write_csv(fbs, outputs["fbs"])
    write_csv(ghi, outputs["ghi"])
    write_csv(panel, outputs["panel"])
    write_csv(coverage, outputs["coverage"])
    write_csv(country_coverage, outputs["country_coverage"])

    summary = {
        "artifact": "official outcome and benchmark acquisition",
        "snapshot_date": args.snapshot_date,
        "accessed_at_utc": accessed_at,
        "sources": {
            "faostat_food_security_catalog": FAOSTAT_FS_CATALOG,
            "faostat_food_balances_catalog": FAOSTAT_FBS_CATALOG,
            "ghi_methodology": GHI_METHODOLOGY_URL if args.include_ghi else None,
        },
        "download_count": len(downloads),
        "coverage": coverage.replace({np.nan: None}).to_dict(orient="records"),
        "validation": {
            "target_countries": 31,
            "pou_rows": int(len(pou)),
            "pou_published_rows": int((~pou["value_status"].eq("absent_from_api_response")).sum()),
            "pou_exact_rows": int(pou["exact_value"].notna().sum()),
            "pou_left_censored_rows": int(pou["value_status"].eq("left_censored").sum()),
            "fies_rows": int(len(fies)),
            "fies_exact_rows": int(fies["exact_value"].notna().sum()),
            "fies_suppressed_rows": int(fies["value_status"].eq("suppressed").sum()),
            "fies_absent_rows": int(fies["value_status"].eq("absent_from_api_response").sum()),
            "cidr_exact_rows": int(cidr["exact_value"].notna().sum()),
            "fbs_ssr_exact_rows": int(fbs["fbs_cereal_ssr_candidate_pct"].notna().sum()),
            "ghi_published_rows": int((~ghi["value_status"].eq("absent_from_api_response")).sum()),
            "ghi_left_censored_rows": int(ghi["value_status"].eq("left_censored").sum()),
            "panel_rows": int(len(panel)),
            "panel_unique_keys": int(panel[["iso3", "year"]].drop_duplicates().shape[0]),
            "country_coverage_rows": int(len(country_coverage)),
            "point_imputation_performed": False,
        },
        "scope_decision": {
            "allowed": [
                "PoU censor-aware preprocessing on all 31 countries",
                "FIES unbalanced validation on published country-periods",
                "30-country modern FBS cereal SSR sensitivity",
                "unbalanced official CIDR/self-sufficiency sensitivity",
                *( ["24-country four-point GHI benchmark"] if args.include_ghi else [] ),
            ],
            "not_allowed": [
                "replace censor thresholds with point values without sensitivity analysis",
                "impute suppressed FIES values",
                "synthesize GHI for seven absent countries",
                "claim a balanced complete outcome panel",
            ],
        },
    }
    summary_path = processed_root / "summary.json"
    write_json_atomic(summary_path, summary)

    output_records = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in [*outputs.values(), summary_path]
    ]
    manifest = {
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "downloads": downloads,
        "outputs": output_records,
    }
    manifest_path = raw_root / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    processed_manifest_path = processed_root / "manifest.json"
    write_json_atomic(processed_manifest_path, manifest)

    print(
        json.dumps(
            {
                "raw_root": str(raw_root),
                "processed_root": str(processed_root),
                "downloads": len(downloads),
                "pou_exact": int(pou["exact_value"].notna().sum()),
                "pou_censored": int(pou["value_status"].eq("left_censored").sum()),
                "fies_exact": int(fies["exact_value"].notna().sum()),
                "fies_suppressed": int(fies["value_status"].eq("suppressed").sum()),
                "cidr_exact": int(cidr["exact_value"].notna().sum()),
                "fbs_ssr_exact": int(fbs["fbs_cereal_ssr_candidate_pct"].notna().sum()),
                "ghi_published": int((~ghi["value_status"].eq("absent_from_api_response")).sum()),
                "panel_rows": len(panel),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
