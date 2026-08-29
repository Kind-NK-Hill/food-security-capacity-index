from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import legacy_weight_audit as legacy  # noqa: E402
import run_analysis as core  # noqa: E402


class DummyRegressor:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return values[:, 0] + 10.0 * values[:, 2]


class RevisedPipelineHelperTests(unittest.TestCase):
    def test_simplex_is_complete_and_matches_r_traversal(self) -> None:
        grid = legacy.simplex_grid()
        self.assertEqual(len(grid), 5151)
        self.assertTrue(
            np.allclose(
                grid[["w1_index_1", "w2_index_2", "w3_one_minus_ghi"]].sum(axis=1),
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
        self.assertEqual(
            grid.iloc[0][
                ["w1_index_1", "w2_index_2", "w3_one_minus_ghi"]
            ].tolist(),
            [0.0, 0.0, 1.0],
        )
        self.assertEqual(
            grid.iloc[100][
                ["w1_index_1", "w2_index_2", "w3_one_minus_ghi"]
            ].tolist(),
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(
            grid.iloc[101][
                ["w1_index_1", "w2_index_2", "w3_one_minus_ghi"]
            ].tolist(),
            [0.0, 0.01, 0.99],
        )
        fixed = grid.loc[
            np.isclose(grid["w1_index_1"], 0.19)
            & np.isclose(grid["w2_index_2"], 0.15)
            & np.isclose(grid["w3_one_minus_ghi"], 0.66)
        ]
        self.assertEqual(len(fixed), 1)

    def test_tv_uses_r_right_closed_internal_breaks(self) -> None:
        # With breaks from 0.15 to 0.75, 0.21 is the first internal break.
        # R's default right-closed histogram puts both observations in bin 1.
        value = legacy.original_r_histogram_tv(np.array([0.15, 0.21]))
        self.assertAlmostEqual(value, 0.9, places=12)

    def test_fold_preprocessing_keeps_indicator_when_train_is_complete(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 11.0, 12.0]})
        test = pd.DataFrame({"a": [np.nan, 4.0], "b": [13.0, 14.0]})
        train_matrix, test_matrix, names, _, _ = core.fit_fold_preprocessing(
            train,
            test,
            features=["a", "b"],
            missing_indicator_features=["a"],
            standardize=False,
        )
        self.assertEqual(names, ["a", "b", "missingindicator_a"])
        np.testing.assert_array_equal(train_matrix[:, 2], np.zeros(3))
        np.testing.assert_array_equal(test_matrix[:, 2], np.array([1.0, 0.0]))
        self.assertEqual(test_matrix[0, 0], 2.0)

    def test_grouped_permutation_moves_value_and_indicator_together(self) -> None:
        values = np.array(
            [
                [1.0, 4.0, 0.0],
                [2.0, 3.0, 0.0],
                [1.5, 2.0, 1.0],
                [4.0, 1.0, 0.0],
            ]
        )
        outcome = DummyRegressor().predict(values)
        result = core.grouped_permutation_mae_importance(
            DummyRegressor(),
            values,
            outcome,
            features=["a", "b"],
            transformed_feature_names=["a", "b", "missingindicator_a"],
            missing_indicator_features=["a"],
            random_state=42,
            repeats=10,
        )
        group = result.set_index("feature").loc["a", "permutation_group_columns"]
        self.assertEqual(group, "a;missingindicator_a")
        self.assertGreater(
            result.set_index("feature").loc[
                "a", "grouped_permutation_importance_mean_mae_increase"
            ],
            0.0,
        )

    def test_manifest_and_coverage_gate_are_recomputed(self) -> None:
        rows: list[dict[str, object]] = []
        for country_index in range(30):
            for year in range(2010, 2022):
                row: dict[str, object] = {
                    "iso3": f"C{country_index:02d}",
                    "country": f"Country {country_index:02d}",
                    "year": year,
                    "legacy_category": "unclassified",
                    core.TARGET: 0.5,
                }
                for feature_index, feature in enumerate(core.FEATURES):
                    row[feature] = float(country_index + year + feature_index)
                rows.append(row)
        panel = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            panel_path = root / "explanatory_panel.csv"
            summary_path = root / "summary.json"
            manifest_path = root / "manifest.json"
            panel.to_csv(panel_path, index=False)
            summary_path.write_text(
                json.dumps(
                    {
                        "data_gate": {
                            "modeling_allowed": True,
                            "passed_features": core.FEATURES,
                        }
                    }
                ),
                encoding="utf-8",
            )

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest().upper()

            manifest_path.write_text(
                json.dumps(
                    {
                        "outputs": [
                            {"path": str(panel_path), "sha256": digest(panel_path)},
                            {"path": str(summary_path), "sha256": digest(summary_path)},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            retained = core.validate_explanatory_gate_and_manifest(
                panel, panel_path, summary_path, manifest_path
            )
            self.assertEqual(retained, core.FEATURES)

            panel.loc[0:200, core.FEATURES[0]] = np.nan
            with self.assertRaisesRegex(ValueError, "disagree on the data gate"):
                core.validate_explanatory_gate_and_manifest(
                    panel, panel_path, summary_path, manifest_path
                )

    def test_external_alignment_joins_on_iso3_year_and_checks_names(self) -> None:
        capacity_rows: list[dict[str, object]] = []
        outcome_rows: list[dict[str, object]] = []
        for index in range(20):
            iso3 = f"C{index:02d}"
            country = f"Country {index:02d}"
            capacity_rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "year": 2016,
                    core.TARGET: index / 19.0,
                }
            )
            outcome = 20.0 - index
            outcome_rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "year": 2016,
                    "pou_year_period": "2015-2017",
                    "pou_lower_bound": outcome,
                    "pou_upper_bound": outcome,
                    "pou_value_status": "observed_exact",
                    "fies_year_period": "2015-2017",
                    "fies_exact_value": outcome,
                    "fies_value_status": "observed_exact",
                    "ghi_lower_bound": outcome,
                    "ghi_upper_bound": outcome,
                    "ghi_value_status": "observed_exact",
                }
            )
        capacity = pd.DataFrame(capacity_rows)
        outcomes = pd.DataFrame(outcome_rows)
        summary, detail, source = core.external_alignment(capacity, outcomes)
        self.assertEqual(len(summary), 5)
        self.assertEqual(len(detail), 100)
        self.assertEqual(len(source), 100)
        self.assertTrue(summary["direction_consistent"].all())

        no_ghi = outcomes.drop(
            columns=["ghi_lower_bound", "ghi_upper_bound", "ghi_value_status"]
        )
        summary_no_ghi, detail_no_ghi, source_no_ghi = core.external_alignment(
            capacity, no_ghi
        )
        self.assertEqual(len(summary_no_ghi), 3)
        self.assertEqual(len(detail_no_ghi), 60)
        self.assertEqual(len(source_no_ghi), 60)

        empty_ghi = outcomes.copy()
        empty_ghi[["ghi_lower_bound", "ghi_upper_bound"]] = np.nan
        empty_ghi["ghi_value_status"] = "absent_from_api_response"
        summary_empty_ghi, _, _ = core.external_alignment(capacity, empty_ghi)
        self.assertEqual(len(summary_empty_ghi), 3)

        outcomes.loc[0, "country"] = "Wrong display name"
        with self.assertRaisesRegex(ValueError, "display-name mismatch"):
            core.external_alignment(capacity, outcomes)


if __name__ == "__main__":
    unittest.main()
