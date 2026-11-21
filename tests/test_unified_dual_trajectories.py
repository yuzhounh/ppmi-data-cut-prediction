"""Synthetic unit tests for the unified PPMI dual-trajectory pipeline (no private data required)."""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import openpyxl

_HERE = Path(__file__).resolve().parent
if _HERE.name == "tests":
    _CODE_ROOT = _HERE.parent
else:
    _CODE_ROOT = _HERE
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import build_unified_manuscript_assets as assets
import ppmi_task_registry as reg
import run_unified_dual_trajectories as runner


def _write_synthetic_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_xlsx(path: Path, sheet_name: str, headers: list[str], rows: list[dict]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    wb.save(path)
    wb.close()


class TestUnifiedDualTrajectories(unittest.TestCase):
    def test_resolve_data_dir_priority(self) -> None:
        marker = reg.extract_filename(reg.REFERENCE)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli_dir = root / "cli_data"
            env_dir = root / "env_data" / "curated"
            cli_dir.mkdir(parents=True)
            env_dir.mkdir(parents=True)
            (cli_dir / marker).write_text("stub", encoding="utf-8")
            (env_dir / marker).write_text("stub", encoding="utf-8")

            with mock.patch.dict(os.environ, {"PPMI_DATA_DIR": str(root / "env_data")}):
                self.assertEqual(reg.resolve_data_dir(cli_dir), cli_dir)
                self.assertEqual(reg.resolve_data_dir(None), env_dir)

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PPMI_DATA_DIR", None)
                with self.assertRaises(FileNotFoundError):
                    reg.resolve_data_dir(root / "missing")

    def test_unified_scheduled_visit_filter_and_window_closure(self) -> None:
        headers = ["PATNO", "COHORT", "EVENT_ID", "visit_date", "age", "moca", "cogstate"]
        # Snapshot 20240129 -> cutoff = 2024*12 + 1 - 2 = 2023-12
        # Baseline 2021-06 -> +24m = 2023-06, +27m = 2023-09 <= 2023-12 (window closed)
        # Baseline 2022-01 -> +27m = 2024-04 > 2023-12 (window NOT closed)
        rows = [
            # Participant 101: unscheduled U01 at exact 24m must be ignored; V06 at 23m and V07 at 25m -> tie picks earlier 23m
            {"PATNO": "101", "COHORT": 1, "EVENT_ID": "BL", "visit_date": "06/2021", "age": 62.5, "moca": 27, "cogstate": 1},
            {"PATNO": "101", "COHORT": 1, "EVENT_ID": "U01", "visit_date": "06/2023", "age": 64.5, "moca": 10, "cogstate": 2},
            {"PATNO": "101", "COHORT": 1, "EVENT_ID": "V06", "visit_date": "05/2023", "age": 64.4, "moca": 25, "cogstate": 1},
            {"PATNO": "101", "COHORT": 1, "EVENT_ID": "V07", "visit_date": "07/2023", "age": 64.6, "moca": 21, "cogstate": 2},
            # Participant 102: window not closed by cutoff 2023-12
            {"PATNO": "102", "COHORT": 1, "EVENT_ID": "BL", "visit_date": "01/2022", "age": 59.0, "moca": 28, "cogstate": 1},
            {"PATNO": "102", "COHORT": 1, "EVENT_ID": "V06", "visit_date": "11/2023", "age": 60.8, "moca": 26, "cogstate": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _write_synthetic_xlsx(data_dir / reg.extract_filename("20240129"), "20240129", headers, rows)
            hdrs, people, cutoff = reg.people_for(data_dir, "20240129", set(headers))
            t2 = next(t for t in reg.ALL_CANDIDATE_TASKS if t.id == "T2")
            status, base_ids, valid, _ = reg.collect(t2, hdrs, people, cutoff)
            self.assertEqual(status, "available")
            self.assertEqual(base_ids, {"101", "102"})
            # Only 101 has closed 27m window; selected visit is 23m (V06, moca=25), ignoring U01 and preferring earlier tie
            self.assertEqual(set(valid.keys()), {"101"})
            self.assertEqual(valid["101"][0], 25.0)
            self.assertEqual(people["101"][2]["visit_date"], reg.parse_month("05/2023"))

    def test_unified_stratified_reserve_is_deterministic_and_balanced(self) -> None:
        versions = ("20230612", "20240129", "20260810")
        cohorts = {
            "20230612": {f"P{i}": (1 if i < 100 else 2) for i in range(150)},
            "20240129": {f"P{i}": (1 if i < 100 else (2 if i < 150 else 4)) for i in range(250)},
            "20260810": {f"P{i}": (1 if i < 100 else (2 if i < 150 else 4)) for i in range(300)},
        }
        reserved_a, meta_a = reg.unified_reserve(versions, cohorts)
        reserved_b, meta_b = reg.unified_reserve(versions, cohorts)
        self.assertEqual(reserved_a, reserved_b)
        self.assertEqual(meta_a, meta_b)
        # Only participants in >=2 contiguous cuts ending at reference (250 people) are eligible; P250..P299 excluded
        self.assertEqual(meta_a["eligible_pool"], 250)
        self.assertEqual(meta_a["total_reserved"], 75)  # 30% of 100 PD + 30% of 50 HC + 30% of 100 Prodromal
        self.assertTrue(all(int(pid[1:]) < 250 for pid in reserved_a))

    def test_records_match_age_tolerance_only(self) -> None:
        ref = runner.TrainingRecord(24200, 24224, (65.12345678, 26.0, None), 25.0)
        # Small age rounding difference (<=0.01 yr) matches
        rounded = runner.TrainingRecord(24200, 24224, (65.1235, 26.0, None), 25.0)
        self.assertTrue(runner.records_match(rounded, ref, age_index=0))
        # True birthdate/visit revision (>0.01 yr) fails
        shifted_age = runner.TrainingRecord(24200, 24224, (65.15, 26.0, None), 25.0)
        self.assertFalse(runner.records_match(shifted_age, ref, age_index=0))
        # Non-age predictor mismatch fails even if tiny
        non_age_diff = runner.TrainingRecord(24200, 24224, (65.1235, 26.001, None), 25.0)
        self.assertFalse(runner.records_match(non_age_diff, ref, age_index=0))
        # Missingness pattern mismatch fails
        miss_diff = runner.TrainingRecord(24200, 24224, (65.1235, 26.0, 1.0), 25.0)
        self.assertFalse(runner.records_match(miss_diff, ref, age_index=0))
        # Outcome or month mismatch fails
        outcome_diff = runner.TrainingRecord(24200, 24224, (65.1235, 26.0, None), 24.0)
        self.assertFalse(runner.records_match(outcome_diff, ref, age_index=0))

    def test_halving_and_tie_permutation_invariants(self) -> None:
        self.assertEqual(runner.halving(800), (800, 400, 200, 100, 50, 25))
        cut_rank = np.array([0] * 40 + [1] * 25 + [3] * 15, dtype=int)
        order_0 = runner.tie_permutation(cut_rank, "T2", repeat=0)
        order_1 = runner.tie_permutation(cut_rank, "T2", repeat=1)
        # Strictly preserves snapshot order while permuting within each tied snapshot block
        np.testing.assert_array_equal(cut_rank[order_0], cut_rank)
        np.testing.assert_array_equal(cut_rank[order_1], cut_rank)
        self.assertEqual(len(np.unique(order_0)), len(cut_rank))
        self.assertFalse(np.array_equal(order_0, order_1))
        # All records from earlier snapshot 0 always occupy indices 0..39
        self.assertEqual(set(order_0[:40]), set(range(40)))
        self.assertEqual(set(order_0[:65]), set(range(65)))

    def test_predict_matches_sklearn_pipeline_with_missingness(self) -> None:
        rng = np.random.default_rng(12345)
        x_train = rng.normal(size=(120, 4))
        x_eval = rng.normal(size=(40, 4))
        # Introduce missingness in both train and eval
        x_train[rng.random(x_train.shape) < 0.15] = np.nan
        x_eval[rng.random(x_eval.shape) < 0.15] = np.nan
        # Keep col 0 complete so no row is all-NaN
        x_train[:, 0] = rng.normal(60.0, 8.0, size=120)
        x_eval[:, 0] = rng.normal(60.0, 8.0, size=40)

        # Binary classification
        y_bin = (rng.random(120) < 0.35).astype(int)
        p_fast = runner.predict(x_train, y_bin, x_eval, num_classes=2)
        p_ref = runner.reference_predict(x_train, y_bin, x_eval, num_classes=2)
        self.assertLess(float(np.max(np.abs(p_fast - p_ref))), runner.REFERENCE_TOLERANCE)

        # Multinomial 3-class classification
        y_tri = rng.integers(0, 3, size=120)
        p3_fast = runner.predict(x_train, y_tri, x_eval, num_classes=3)
        p3_ref = runner.reference_predict(x_train, y_tri, x_eval, num_classes=3)
        self.assertLess(float(np.max(np.abs(p3_fast - p3_ref))), runner.REFERENCE_TOLERANCE)

        # Continuous ridge regression
        y_reg = 25.0 - 0.1 * x_train[:, 0] + rng.normal(0.0, 2.0, size=120)
        r_fast = runner.predict(x_train, y_reg, x_eval, num_classes=0)
        r_ref = runner.reference_predict(x_train, y_reg, x_eval, num_classes=0)
        self.assertLess(float(np.max(np.abs(r_fast - r_ref))), runner.REFERENCE_TOLERANCE)

    def test_case_loss_definitions(self) -> None:
        # Binary Brier
        y_b = np.array([0, 1])
        pred_b = np.array([[0.8, 0.2], [0.3, 0.7]])
        np.testing.assert_allclose(runner.case_loss(y_b, pred_b, 2), [0.04, 0.09])
        # Multiclass sum-of-squares Brier
        y_m = np.array([0, 2])
        pred_m = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        expected_m = [(0.7 - 1) ** 2 + 0.2**2 + 0.1**2, 0.1**2 + 0.3**2 + (0.6 - 1) ** 2]
        np.testing.assert_allclose(runner.case_loss(y_m, pred_m, 3), expected_m)
        # Regression MAE
        np.testing.assert_allclose(runner.case_loss(np.array([10.0, 20.0]), np.array([12.5, 17.0]), 0), [2.5, 3.0])

    def test_synthetic_csv_and_xlsx_end_to_end_asset_generation(self) -> None:
        # Verify aggregate asset builder handles both multi-tier and single-tier (C05-like) tasks cleanly
        import json
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            res_dir = Path(tmp) / "results"
            fig_dir = Path(tmp) / "figures"
            res_dir.mkdir()
            tasks_df = pd.DataFrame(
                [
                    {
                        "task": "T2",
                        "family": "Cognition",
                        "title": "Two-year MoCA",
                        "timing": "M24",
                        "versions": 3,
                        "first_version": "20230612",
                        "latest_version": "20260810",
                        "evaluation_n": 50,
                        "evaluation_classes": "{}",
                        "primary_metric": "mae",
                        "features": json.dumps(["age", "moca"]),
                        "dev_missingness_pct": json.dumps({"age": 0.0, "moca": 2.0}),
                        "eval_missingness_pct": json.dumps({"age": 0.0, "moca": 0.0}),
                    },
                    {
                        "task": "F02",
                        "family": "Motor and function",
                        "title": "Two-year MDS-UPDRS IV",
                        "timing": "M24",
                        "versions": 3,
                        "first_version": "20230612",
                        "latest_version": "20260810",
                        "evaluation_n": 50,
                        "evaluation_classes": "{}",
                        "primary_metric": "mae",
                        "features": json.dumps(["age", "updrs4_score"]),
                        "dev_missingness_pct": json.dumps({"age": 0.0, "updrs4_score": 80.0}),
                        "eval_missingness_pct": json.dumps({"age": 0.0, "updrs4_score": 75.0}),
                    },
                    {
                        "task": "C05",
                        "family": "Enrollment groups",
                        "title": "Baseline PD vs SWEDD enrollment",
                        "timing": "BL",
                        "versions": 3,
                        "first_version": "20240129",
                        "latest_version": "20260810",
                        "evaluation_n": 50,
                        "evaluation_classes": json.dumps({"PD": 40, "SWEDD": 10}),
                        "primary_metric": "brier",
                        "features": json.dumps(list(reg.CORE)),
                        "dev_missingness_pct": json.dumps({c: 0.0 for c in reg.CORE}),
                        "eval_missingness_pct": json.dumps({c: 0.0 for c in reg.CORE}),
                    },
                ]
            )
            versions_df = pd.DataFrame(
                [
                    {"task": "T2", "version": "20230612", "training_n": 100, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 2.10},
                    {"task": "T2", "version": "20250321", "training_n": 150, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 2.02},
                    {"task": "T2", "version": "20260810", "training_n": 200, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 1.95},
                    {"task": "F02", "version": "20230612", "training_n": 100, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 1.40},
                    {"task": "F02", "version": "20250321", "training_n": 150, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 1.45},
                    {"task": "F02", "version": "20260810", "training_n": 200, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 1.50},
                    {"task": "C05", "version": "20240129", "training_n": 100, "evaluation_n": 50, "primary_metric": "brier", "observed_release_error": 0.050},
                    {"task": "C05", "version": "20250321", "training_n": 150, "evaluation_n": 50, "primary_metric": "brier", "observed_release_error": 0.049},
                    {"task": "C05", "version": "20260810", "training_n": 200, "evaluation_n": 50, "primary_metric": "brier", "observed_release_error": 0.048},
                ]
            )
            size_df = pd.DataFrame(
                [
                    {"task": "T2", "primary_metric": "mae", "evaluation_n": 50, "training_n": 50, "verified_cut": "20230612", "observed_error": 2.20, "error_p025": 2.00, "error_p975": 2.40, "valid_rounds": 1000, "boundary_group_split": True, "status": "fit"},
                    {"task": "T2", "primary_metric": "mae", "evaluation_n": 50, "training_n": 100, "verified_cut": "20230612", "observed_error": 2.05, "error_p025": 1.98, "error_p975": 2.12, "valid_rounds": 1000, "boundary_group_split": True, "status": "fit"},
                    {"task": "T2", "primary_metric": "mae", "evaluation_n": 50, "training_n": 200, "verified_cut": "20260810", "observed_error": 1.95, "error_p025": 1.95, "error_p975": 1.95, "valid_rounds": 1000, "boundary_group_split": False, "status": "fit"},
                    {"task": "F02", "primary_metric": "mae", "evaluation_n": 50, "training_n": 100, "verified_cut": "20230612", "observed_error": 1.45, "error_p025": 1.35, "error_p975": 1.55, "valid_rounds": 1000, "boundary_group_split": True, "status": "fit"},
                    {"task": "F02", "primary_metric": "mae", "evaluation_n": 50, "training_n": 200, "verified_cut": "20260810", "observed_error": 1.50, "error_p025": 1.50, "error_p975": 1.50, "valid_rounds": 1000, "boundary_group_split": False, "status": "fit"},
                    {"task": "C05", "primary_metric": "brier", "evaluation_n": 50, "training_n": 100, "verified_cut": "20230612", "observed_error": np.nan, "error_p025": np.nan, "error_p975": np.nan, "valid_rounds": 0, "boundary_group_split": True, "status": "missing_training_class"},
                    {"task": "C05", "primary_metric": "brier", "evaluation_n": 50, "training_n": 200, "verified_cut": "20260810", "observed_error": 0.048, "error_p025": 0.048, "error_p975": 0.048, "valid_rounds": 1000, "boundary_group_split": False, "status": "fit"},
                ]
            )
            tasks_df.to_csv(res_dir / "task_index.csv", index=False)
            versions_df.to_csv(res_dir / "real_version_trajectory.csv", index=False)
            size_df.to_csv(res_dir / "chronological_size_trajectory.csv", index=False)

            t, v, s = assets.load_inputs(res_dir)
            desc = assets.describe_trajectories(t, v, s)
            self.assertEqual(desc.loc[desc.task == "T2", "direction"].iloc[0], "Declining")
            self.assertEqual(desc.loc[desc.task == "C05", "direction"].iloc[0], "Single tier")
            assets.make_tables(desc, s, res_dir)
            assets.make_task_catalog(t, res_dir)
            assets.make_figures(t, v, s, res_dir, fig_dir, ["T2", "F02", "C05"])
            self.assertTrue((res_dir / "paper_family_table.tex").is_file())
            self.assertTrue((res_dir / "paper_task_catalog.tex").is_file())
            self.assertTrue((res_dir / "paper_task_results.tex").is_file())
            self.assertTrue((fig_dir / "chronological_examples.pdf").is_file())
            self.assertTrue((res_dir / "paper_all_trajectories.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
