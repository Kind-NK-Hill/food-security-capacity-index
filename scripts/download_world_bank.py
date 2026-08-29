from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


COUNTRIES = [
    ("ARG", "Argentina"),
    ("BGD", "Bangladesh"),
    ("BRA", "Brazil"),
    ("CHN", "China"),
    ("COL", "Colombia"),
    ("COD", "Congo, Dem. Rep."),
    ("EGY", "Egypt, Arab Rep."),
    ("ETH", "Ethiopia"),
    ("FRA", "France"),
    ("DEU", "Germany"),
    ("IND", "India"),
    ("IDN", "Indonesia"),
    ("IRN", "Iran, Islamic Rep."),
    ("ITA", "Italy"),
    ("JPN", "Japan"),
    ("KEN", "Kenya"),
    ("KOR", "Korea, Rep."),
    ("MYS", "Malaysia"),
    ("MEX", "Mexico"),
    ("MMR", "Myanmar"),
    ("NGA", "Nigeria"),
    ("PAK", "Pakistan"),
    ("PHL", "Philippines"),
    ("RUS", "Russian Federation"),
    ("ZAF", "South Africa"),
    ("TZA", "Tanzania"),
    ("THA", "Thailand"),
    ("TUR", "Turkiye"),
    ("GBR", "United Kingdom"),
    ("USA", "United States"),
    ("VNM", "Viet Nam"),
]

INDICATORS = {
    "NV.AGR.TOTL.ZS": {
        "role": "legacy_index_1",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "NV.MNF.FBTO.ZS.UN": {
        "role": "legacy_index_1",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "NY.GDP.MKTP.CD": {
        "role": "legacy_index_1",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "TX.VAL.FOOD.ZS.UN": {
        "role": "legacy_index_1",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "TX.VAL.MRCH.CD.WT": {
        "role": "legacy_index_1",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "TM.VAL.FOOD.ZS.UN": {
        "role": "legacy_index_1",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "TM.VAL.MRCH.CD.WT": {
        "role": "legacy_index_1",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "AG.PRD.CREL.MT": {
        "role": "legacy_index_2",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "SP.POP.TOTL": {
        "role": "legacy_index_2",
        "audit_start": 1994,
        "audit_end": 2021,
    },
    "SN.ITK.DEFC.ZS": {
        "role": "outcome_pou",
        "audit_start": 2001,
        "audit_end": 2023,
    },
    "SN.ITK.MSFI.ZS": {
        "role": "outcome_fies_moderate_or_severe",
        "audit_start": 2015,
        "audit_end": 2023,
    },
}

REQUEST_START = 1994
REQUEST_END = 2025
WORLD_BANK_API = "https://api.worldbank.org/v2"
SOURCE_PAGE = "https://data.worldbank.org/"
USER_AGENT = "FoodSecurityRebuildPilot/1.0 (reproducible academic data audit)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize World Bank inputs for the cereal-capacity project."
    )
    parser.add_argument(
        "--snapshot-date",
        default=date.today().isoformat(),
        help="Version folder in YYYY-MM-DD form (default: today's date).",
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
    with temporary.open("wb") as handle:
        handle.write(content)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    write_bytes_atomic(path, content)


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        return response.read()


def data_url(indicator: str) -> str:
    country_path = ";".join(iso for iso, _ in COUNTRIES)
    query = urllib.parse.urlencode(
        {
            "date": f"{REQUEST_START}:{REQUEST_END}",
            "format": "json",
            "per_page": "20000",
        }
    )
    return f"{WORLD_BANK_API}/country/{country_path}/indicator/{indicator}?{query}"


def metadata_url(indicator: str) -> str:
    return f"{WORLD_BANK_API}/indicator/{indicator}?format=json"


def parse_api_payload(content: bytes, url: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected World Bank response for {url}")
    page_metadata = payload[0]
    records = payload[1] or []
    if page_metadata.get("page") != 1:
        raise RuntimeError(f"Unexpected pagination for {url}: {page_metadata}")
    if page_metadata.get("pages") != 1:
        raise RuntimeError(f"Response is paginated despite per_page=20000 for {url}")
    return page_metadata, records


def parse_indicator_metadata(content: bytes, url: str) -> dict[str, Any]:
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise RuntimeError(f"Unexpected indicator metadata response for {url}")
    return payload[1][0]


def main() -> int:
    args = parse_args()
    datetime.strptime(args.snapshot_date, "%Y-%m-%d")

    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw" / "world_bank" / args.snapshot_date
    processed_root = project_root / "data" / "processed" / "world_bank" / args.snapshot_date
    if raw_root.exists() or processed_root.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing World Bank snapshot. "
            "Use a new --snapshot-date or move the existing directories."
        )
    accessed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    requested_names = dict(COUNTRIES)
    requested_isos = [iso for iso, _ in COUNTRIES]
    expected_years = list(range(REQUEST_START, REQUEST_END + 1))

    all_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    request_manifest: list[dict[str, Any]] = []
    page_lastupdated_values: set[str] = set()

    for indicator, config in INDICATORS.items():
        indicator_data_url = data_url(indicator)
        indicator_metadata_url = metadata_url(indicator)
        data_content = fetch(indicator_data_url)
        metadata_content = fetch(indicator_metadata_url)

        data_path = raw_root / "indicator" / f"{indicator}.data.json"
        metadata_path = raw_root / "metadata" / f"{indicator}.metadata.json"
        write_bytes_atomic(data_path, data_content)
        write_bytes_atomic(metadata_path, metadata_content)

        page_metadata, records = parse_api_payload(data_content, indicator_data_url)
        indicator_metadata = parse_indicator_metadata(metadata_content, indicator_metadata_url)
        lastupdated = str(page_metadata.get("lastupdated", ""))
        page_lastupdated_values.add(lastupdated)

        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for record in records:
            iso3 = record.get("countryiso3code")
            year_raw = record.get("date")
            if iso3 not in requested_names or year_raw is None:
                continue
            by_key[(iso3, int(year_raw))] = record

        for iso3 in requested_isos:
            for year in expected_years:
                record = by_key.get((iso3, year), {})
                value = record.get("value")
                all_rows.append(
                    {
                        "indicator_code": indicator,
                        "indicator_name": indicator_metadata.get("name", ""),
                        "role": config["role"],
                        "iso3": iso3,
                        "requested_country_name": requested_names[iso3],
                        "api_country_name": (record.get("country") or {}).get("value", ""),
                        "year": year,
                        "value": "" if value is None else value,
                        "is_missing": 1 if value is None else 0,
                        "unit": indicator_metadata.get("unit", ""),
                        "decimal": record.get("decimal", ""),
                        "obs_status": record.get("obs_status", ""),
                        "wdi_lastupdated": lastupdated,
                        "api_url": indicator_data_url,
                        "accessed_at_utc": accessed_at,
                    }
                )

        source = indicator_metadata.get("source") or {}
        metadata_rows.append(
            {
                "indicator_code": indicator,
                "indicator_name": indicator_metadata.get("name", ""),
                "role": config["role"],
                "unit": indicator_metadata.get("unit", ""),
                "source_id": source.get("id", ""),
                "source_name": source.get("value", ""),
                "source_note": indicator_metadata.get("sourceNote", ""),
                "source_organization": indicator_metadata.get("sourceOrganization", ""),
                "topics": " | ".join(topic.get("value", "") for topic in indicator_metadata.get("topics", [])),
                "metadata_url": indicator_metadata_url,
                "data_url": indicator_data_url,
                "wdi_lastupdated": lastupdated,
                "accessed_at_utc": accessed_at,
            }
        )

        request_manifest.append(
            {
                "indicator_code": indicator,
                "role": config["role"],
                "data_url": indicator_data_url,
                "metadata_url": indicator_metadata_url,
                "raw_data_path": str(data_path.relative_to(project_root)),
                "raw_data_sha256": sha256_bytes(data_content),
                "raw_metadata_path": str(metadata_path.relative_to(project_root)),
                "raw_metadata_sha256": sha256_bytes(metadata_content),
                "wdi_lastupdated": lastupdated,
            }
        )

    all_rows.sort(key=lambda row: (row["indicator_code"], row["iso3"], row["year"]))
    long_fields = [
        "indicator_code",
        "indicator_name",
        "role",
        "iso3",
        "requested_country_name",
        "api_country_name",
        "year",
        "value",
        "is_missing",
        "unit",
        "decimal",
        "obs_status",
        "wdi_lastupdated",
        "api_url",
        "accessed_at_utc",
    ]
    long_path = processed_root / "wdi_31countries_1994_2025_long.csv"
    write_csv_atomic(long_path, long_fields, all_rows)

    legacy_rows = [
        row
        for row in all_rows
        if row["role"] in {"legacy_index_1", "legacy_index_2"}
        and 1994 <= int(row["year"]) <= 2021
    ]
    legacy_path = processed_root / "wdi_legacy_inputs_1994_2021_long.csv"
    write_csv_atomic(legacy_path, long_fields, legacy_rows)

    outcome_rows = [
        row
        for row in all_rows
        if row["role"] in {"outcome_pou", "outcome_fies_moderate_or_severe"}
        and 2001 <= int(row["year"]) <= 2023
    ]
    outcomes_path = processed_root / "wdi_outcomes_2001_2023_long.csv"
    write_csv_atomic(outcomes_path, long_fields, outcome_rows)

    metadata_fields = [
        "indicator_code",
        "indicator_name",
        "role",
        "unit",
        "source_id",
        "source_name",
        "source_note",
        "source_organization",
        "topics",
        "metadata_url",
        "data_url",
        "wdi_lastupdated",
        "accessed_at_utc",
    ]
    metadata_path = processed_root / "wdi_indicator_metadata.csv"
    write_csv_atomic(metadata_path, metadata_fields, metadata_rows)

    country_rows = [
        {"iso3": iso3, "requested_country_name": name} for iso3, name in COUNTRIES
    ]
    countries_path = processed_root / "country_mapping.csv"
    write_csv_atomic(countries_path, ["iso3", "requested_country_name"], country_rows)

    values_by_indicator_country: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in all_rows:
        values_by_indicator_country[row["indicator_code"]][row["iso3"]].append(row)

    coverage_rows: list[dict[str, Any]] = []
    for indicator, config in INDICATORS.items():
        start = int(config["audit_start"])
        end = int(config["audit_end"])
        window_years = end - start + 1
        counts: dict[str, int] = {}
        observed_years: list[int] = []
        floor_2_5_count = 0

        for iso3 in requested_isos:
            rows = [
                row
                for row in values_by_indicator_country[indicator][iso3]
                if start <= int(row["year"]) <= end
            ]
            observed = [row for row in rows if int(row["is_missing"]) == 0]
            counts[iso3] = len(observed)
            observed_years.extend(int(row["year"]) for row in observed)
            floor_2_5_count += sum(
                1
                for row in observed
                if isinstance(row["value"], (int, float)) and float(row["value"]) == 2.5
            )

        missing_countries = [iso3 for iso3 in requested_isos if counts[iso3] == 0]
        partial_countries = [
            f"{iso3}:{window_years - counts[iso3]}"
            for iso3 in requested_isos
            if 0 < counts[iso3] < window_years
        ]
        coverage_rows.append(
            {
                "indicator_code": indicator,
                "role": config["role"],
                "audit_start": start,
                "audit_end": end,
                "expected_country_years": len(COUNTRIES) * window_years,
                "nonmissing_country_years": sum(counts.values()),
                "missing_country_years": len(COUNTRIES) * window_years - sum(counts.values()),
                "countries_with_any_value": sum(1 for count in counts.values() if count > 0),
                "countries_complete_in_window": sum(
                    1 for count in counts.values() if count == window_years
                ),
                "missing_countries": ",".join(missing_countries),
                "partial_countries_iso3_missing_count": ",".join(partial_countries),
                "first_observed_year": min(observed_years) if observed_years else "",
                "last_observed_year": max(observed_years) if observed_years else "",
                "value_2_5_count": floor_2_5_count,
                "wdi_lastupdated": next(
                    row["wdi_lastupdated"]
                    for row in all_rows
                    if row["indicator_code"] == indicator
                ),
                "accessed_at_utc": accessed_at,
            }
        )

    coverage_fields = [
        "indicator_code",
        "role",
        "audit_start",
        "audit_end",
        "expected_country_years",
        "nonmissing_country_years",
        "missing_country_years",
        "countries_with_any_value",
        "countries_complete_in_window",
        "missing_countries",
        "partial_countries_iso3_missing_count",
        "first_observed_year",
        "last_observed_year",
        "value_2_5_count",
        "wdi_lastupdated",
        "accessed_at_utc",
    ]
    coverage_path = processed_root / "wdi_coverage_summary.csv"
    write_csv_atomic(coverage_path, coverage_fields, coverage_rows)

    generated_files = [
        long_path,
        legacy_path,
        outcomes_path,
        metadata_path,
        countries_path,
        coverage_path,
    ]
    manifest = {
        "snapshot_date": args.snapshot_date,
        "accessed_at_utc": accessed_at,
        "world_bank_api_base": WORLD_BANK_API,
        "source_page": SOURCE_PAGE,
        "request_year_range": [REQUEST_START, REQUEST_END],
        "country_count": len(COUNTRIES),
        "indicator_count": len(INDICATORS),
        "normalized_grid_row_count": len(all_rows),
        "wdi_lastupdated_values": sorted(page_lastupdated_values),
        "python_version": sys.version,
        "platform": platform.platform(),
        "script_path": str(Path(__file__).resolve().relative_to(project_root)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "requests": request_manifest,
        "generated_files": [
            {
                "path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated_files
        ],
    }
    manifest_path = raw_root / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    print(f"snapshot_date={args.snapshot_date}")
    print(f"wdi_lastupdated={','.join(sorted(page_lastupdated_values))}")
    print(f"raw_root={raw_root}")
    print(f"processed_root={processed_root}")
    print(f"normalized_rows={len(all_rows)}")
    print(f"legacy_rows={len(legacy_rows)}")
    print(f"outcome_rows={len(outcome_rows)}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
