#!/usr/bin/env python3
"""Idempotent public-release audit for the cereal-capacity repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".tex", ".toml", ".yml", ".yaml", ".json", ".csv", ".txt"}
FORBIDDEN_PARTS = {".venv", "venv", "__pycache__", "source_archive", "rendered_pages", "tmp"}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\b(?:gho_|github_pat_)[A-Za-z0-9_]+"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|auth[_-]?token)\s*[:=]\s*['\"][^'\"]+"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable audit report.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {"check": name, "pass": bool(passed), "actual": json_safe(actual), "expected": expected}
        )

    required = [
        "README.md",
        "LICENSE",
        "DATA_SOURCES.md",
        "THIRD_PARTY_NOTICES.md",
        "requirements.txt",
        "config/reproduction.toml",
        "config/country_categories.csv",
        "data/processed/capacity/capacity_v2_panel.csv",
        "data/processed/explanatory/explanatory_panel.csv",
        "data/processed/outcomes/outcome_panel.csv",
        "results/core/model_metrics.csv",
        "report/Food_Security_Capacity_Index.tex",
        "report/Food_Security_Capacity_Index.pdf",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    check("required public files", not missing, missing, [])

    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        files = [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    else:
        files = [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and not any(part in FORBIDDEN_PARTS for part in path.relative_to(ROOT).parts)
            and path.suffix.lower() not in {".pyc", ".log", ".aux", ".out", ".toc"}
        ]
    forbidden = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if any(part in FORBIDDEN_PARTS for part in path.relative_to(ROOT).parts)
        or path.suffix.lower() == ".pyc"
    ]
    check("no excluded private/build artifacts", not forbidden, forbidden, [])

    oversized = [
        {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size}
        for path in files
        if path.stat().st_size > 50 * 1024 * 1024
    ]
    check("no file exceeds 50 MiB", not oversized, oversized, [])

    machine_paths: list[str] = []
    secrets: list[dict[str, str]] = []
    windows_path = re.compile(r"[A-Za-z]:\\(?:Users|Grad_Study|Obsidian|[A-Za-z0-9_. -]+)\\")
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(ROOT).as_posix()
        if windows_path.search(text):
            machine_paths.append(relative)
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                secrets.append({"path": relative, "pattern": label})
    check("no machine-specific Windows paths", not machine_paths, sorted(set(machine_paths)), [])
    check("no common secret patterns", not secrets, secrets, [])

    capacity = pd.read_csv(ROOT / "data/processed/capacity/capacity_v2_panel.csv")
    explanatory = pd.read_csv(ROOT / "data/processed/explanatory/explanatory_panel.csv")
    outcomes = pd.read_csv(ROOT / "data/processed/outcomes/outcome_panel.csv")
    check(
        "capacity panel contract",
        len(capacity) == 420
        and capacity["iso3"].nunique() == 30
        and not capacity.duplicated(["iso3", "year"]).any()
        and "JPN" not in set(capacity["iso3"]),
        {"rows": len(capacity), "countries": capacity["iso3"].nunique()},
        "420 unique rows, 30 countries, Japan absent",
    )
    feature_names = [
        "electricity_access_pct",
        "rural_population_pct",
        "rural_population_growth_pct",
        "female_agricultural_employment_pct",
        "arable_land_hectares_per_person",
        "fertilizer_consumption_kg_per_hectare_arable_land",
    ]
    check(
        "explanatory panel contract",
        len(explanatory) == 360
        and explanatory["iso3"].nunique() == 30
        and not explanatory.duplicated(["iso3", "year"]).any()
        and not explanatory[feature_names].isna().any().any(),
        {
            "rows": len(explanatory),
            "countries": explanatory["iso3"].nunique(),
            "missing_feature_cells": int(explanatory[feature_names].isna().sum().sum()),
        },
        "360 unique rows, 30 countries, six complete features",
    )
    check(
        "outcome panel contract",
        len(outcomes) == 420
        and outcomes["iso3"].nunique() == 30
        and not outcomes.duplicated(["iso3", "year"]).any()
        and not any(column.startswith("ghi_") for column in outcomes.columns),
        {"rows": len(outcomes), "countries": outcomes["iso3"].nunique()},
        "420 unique PoU/FIES rows and no redistributed GHI columns",
    )

    metrics = pd.read_csv(ROOT / "results/core/model_metrics.csv").set_index("model")
    headline = {
        "baseline_mae": float(metrics.loc["constant_0.50_baseline", "mae"]),
        "ols_r2": float(metrics.loc["ols", "r2"]),
        "rf_mae": float(metrics.loc["shallow_random_forest", "mae"]),
        "rf_r2": float(metrics.loc["shallow_random_forest", "r2"]),
    }
    check(
        "published headline metrics",
        np.isclose(headline["baseline_mae"], 0.229885057471264, atol=1e-12)
        and np.isclose(headline["ols_r2"], -0.230736003417319, atol=1e-12)
        and np.isclose(headline["rf_mae"], 0.191584267877049, atol=1e-12)
        and np.isclose(headline["rf_r2"], 0.319155430260619, atol=1e-12),
        headline,
        "frozen published values",
    )

    figure_names = [
        "method_flow.png",
        "capacity_profile.png",
        "country_structure.png",
        "linear_coefficients.png",
        "rf_shap_beeswarm_appendix.png",
        "legacy_simplex_appendix.png",
    ]
    figure_audit: list[dict[str, Any]] = []
    figures_ok = True
    for name in figure_names:
        path = ROOT / "figures" / name
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            figure_audit.append({"path": f"figures/{name}", "width": width, "height": height})
            figures_ok = figures_ok and width >= 1000 and height >= 500
        except Exception as error:  # pragma: no cover - diagnostic path
            figures_ok = False
            figure_audit.append({"path": f"figures/{name}", "error": str(error)})
    check("six readable report figures", figures_ok, figure_audit, "six PNGs at least 1000x500")

    pdf = ROOT / "report" / "Food_Security_Capacity_Index.pdf"
    pdf_ok = pdf.exists() and pdf.stat().st_size > 500_000 and pdf.read_bytes().startswith(b"%PDF-")
    check(
        "compiled report PDF",
        pdf_ok,
        None if not pdf.exists() else {"bytes": pdf.stat().st_size},
        "valid PDF header and >500 KiB",
    )

    passed = sum(item["pass"] for item in checks)
    report = {
        "status": "pass" if passed == len(checks) else "fail",
        "checks_passed": passed,
        "checks_total": len(checks),
        "failed_checks": [item["check"] for item in checks if not item["pass"]],
        "checks": checks,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
