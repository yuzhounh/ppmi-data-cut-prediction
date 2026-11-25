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

    def test_secondary_metrics_match_sklearn(self) -> None:
        # Binary AUROC with ties
        y_b = np.array([0, 0, 1, 1, 0, 1])
        pred_b = np.array([[0.9, 0.1], [0.6, 0.4], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        got_b = runner.secondary_metrics(y_b, pred_b, 2)
        ref_b = runner.reference_secondary_metrics(y_b, pred_b, 2)
        self.assertLess(abs(got_b[0] - ref_b[0]), 1e-12)
        self.assertTrue(np.isnan(got_b[1]) and np.isnan(got_b[2]))

        # Three-class macro-OvR AUROC with ties
        y_m = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        pred_m = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.5, 0.3, 0.2],
                [0.2, 0.6, 0.2],
                [0.3, 0.4, 0.3],
                [0.1, 0.2, 0.7],
                [0.2, 0.3, 0.5],
                [0.5, 0.3, 0.2],
                [0.2, 0.5, 0.3],
                [0.1, 0.1, 0.8],
            ]
        )
        got_m = runner.secondary_metrics(y_m, pred_m, 3)
        ref_m = runner.reference_secondary_metrics(y_m, pred_m, 3)
        self.assertLess(abs(got_m[0] - ref_m[0]), 1e-12)

        # Regression R^2 and RMSE
        y_r = np.array([10.0, 15.0, 22.0, 30.0, 27.0])
        pred_r = np.array([11.2, 14.5, 20.0, 28.5, 29.0])
        got_r = runner.secondary_metrics(y_r, pred_r, 0)
        ref_r = runner.reference_secondary_metrics(y_r, pred_r, 0)
        self.assertTrue(np.isnan(got_r[0]))
        self.assertLess(abs(got_r[1] - ref_r[1]), 1e-12)
        self.assertLess(abs(got_r[2] - ref_r[2]), 1e-12)

    def test_m1_m2_calibration_auprc_logloss_match_reference(self) -> None:
        rng = np.random.default_rng(77)
        # Binary case with ties, realistic class overlap, and systematic under-prediction
        y_b = np.array([0, 1] * 25)
        logits = np.where(y_b == 1, -0.1, -1.1) + rng.normal(0.0, 0.85, size=len(y_b))
        logits[0] = logits[1]  # induce tie across opposite labels
        p1 = 1.0 / (1.0 + np.exp(-logits))
        pred_b = np.column_stack((1.0 - p1, p1))
        tr_prior = np.array([0.75, 0.25])
        rf_prior = np.array([0.50, 0.50])
        diag_b = runner.diagnostic_metrics(y_b, pred_b, 2, tr_prior, rf_prior, compute_cls_slope=True)
        ref_b = runner.reference_diagnostic_metrics(y_b, pred_b, 2, compute_cls_slope=True)
        self.assertLess(abs(diag_b.auprc - ref_b["auprc"]), 1e-12)
        self.assertLess(abs(diag_b.log_loss - ref_b["log_loss"]), 1e-12)
        self.assertTrue(np.isfinite(diag_b.cal_slope) and np.isfinite(ref_b["cal_slope"]))
        self.assertLess(abs(diag_b.cal_slope - ref_b["cal_slope"]), 1e-6)
        # Verify CITL score equation and sign convention (under-prediction => eo < 1 and citl > 0)
        self.assertLess(diag_b.eo_ratio, 1.0)
        self.assertGreater(diag_b.citl, 0.0)
        eta = np.log(p1 / (1.0 - p1))
        self.assertLess(abs(float(np.mean(1.0 / (1.0 + np.exp(-(eta + diag_b.citl))))) - float(y_b.mean())), 1e-11)
        self.assertGreaterEqual(diag_b.ece, 0.0)
        # Complete separation safely returns NaN instead of diverging
        sep_y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        sep_p = np.array([0.1, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.9])
        self.assertTrue(np.isnan(runner.binary_cal_slope(sep_y, sep_p)))

        # Three-class case with ties
        y_m = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
        pred_m = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.5, 0.3, 0.2],
                [0.2, 0.6, 0.2],
                [0.3, 0.4, 0.3],
                [0.1, 0.2, 0.7],
                [0.2, 0.3, 0.5],
                [0.5, 0.3, 0.2],
                [0.2, 0.5, 0.3],
                [0.1, 0.1, 0.8],
                [0.6, 0.2, 0.2],
                [0.3, 0.5, 0.2],
                [0.2, 0.2, 0.6],
            ]
        )
        diag_m = runner.diagnostic_metrics(y_m, pred_m, 3, np.array([0.4, 0.3, 0.3]), np.array([1 / 3] * 3))
        ref_m = runner.reference_diagnostic_metrics(y_m, pred_m, 3)
        self.assertLess(abs(diag_m.auprc - ref_m["auprc"]), 1e-12)
        self.assertLess(abs(diag_m.log_loss - ref_m["log_loss"]), 1e-12)
        self.assertTrue(np.isnan(diag_m.citl) and np.isfinite(diag_m.macro_abs_citl) and np.isfinite(diag_m.ece))

        # Regression M1 metrics
        y_r = np.array([10.0, 15.0, 22.0, 30.0, 27.0])
        pred_r = np.array([12.0, 16.5, 22.5, 31.0, 29.0])
        diag_r = runner.diagnostic_metrics(y_r, pred_r, 0, train_prior=22.2, ref_prior=float(y_r.mean()))
        self.assertAlmostEqual(diag_r.cal_me, float(np.mean(pred_r - y_r)), places=12)
        self.assertAlmostEqual(diag_r.medae, float(np.median(np.abs(y_r - pred_r))), places=12)
        self.assertAlmostEqual(diag_r.nmae, float(np.mean(np.abs(y_r - pred_r)) / np.std(y_r, ddof=0)), places=12)
        expected_slope = float(np.cov(pred_r, y_r, ddof=0)[0, 1] / np.var(pred_r, ddof=0))
        self.assertAlmostEqual(diag_r.cal_slope, expected_slope, places=12)

    def test_m3_prior_recalibration_identities_and_invariance(self) -> None:
        rng = np.random.default_rng(99)
        y = (rng.random(200) < 0.35).astype(int)
        # True log-odds under 0.35 base rate, then shifted down as if trained on 0.15 base rate
        true_eta = np.where(y == 1, 0.9, -1.5) + rng.normal(0.0, 0.5, size=200)
        delta = np.log((0.15 / 0.85) / (0.35 / 0.65))
        p_shifted = 1.0 / (1.0 + np.exp(-(true_eta + delta)))
        pred_shifted = np.column_stack((1.0 - p_shifted, p_shifted))

        # Identity when train_prior == ref_prior
        same = runner.prior_recalibrate(pred_shifted, np.array([0.65, 0.35]), np.array([0.65, 0.35]), 2)
        np.testing.assert_allclose(same, pred_shifted, atol=1e-14)

        # Exact equivalence to Bayes logit-shift formula for k=2
        adj = runner.prior_recalibrate(pred_shifted, np.array([0.85, 0.15]), np.array([0.65, 0.35]), 2)
        expected_p1 = 1.0 / (1.0 + np.exp(-(np.log(p_shifted / (1.0 - p_shifted)) - delta)))
        np.testing.assert_allclose(adj[:, 1], expected_p1, atol=1e-13)
        # Binary AUROC and AUPRC strictly invariant under monotonic Bayes shift
        self.assertLess(abs(runner.binary_auroc(y, adj[:, 1]) - runner.binary_auroc(y, pred_shifted[:, 1])), 1e-14)
        self.assertLess(abs(runner.binary_auprc(y, adj[:, 1]) - runner.binary_auprc(y, pred_shifted[:, 1])), 1e-14)
        # Prior shift correction strictly lowers Brier and Log-loss here
        self.assertLess(runner.case_loss(y, adj, 2).mean(), runner.case_loss(y, pred_shifted, 2).mean())
        self.assertLess(runner.case_log_loss(y, adj).mean(), runner.case_log_loss(y, pred_shifted).mean())

        # Regression mean-shift alignment leaves slope invariant and lowers MAE under pure mean shift
        y_reg = rng.normal(20.0, 4.0, size=150)
        pred_reg = 0.7 * y_reg + 0.3 * 20.0 + 3.0  # +3 upward training mean shift
        adj_reg = runner.prior_recalibrate(pred_reg, train_prior=23.0, ref_prior=20.0, num_classes=0)
        self.assertLess(np.mean(np.abs(y_reg - adj_reg)), np.mean(np.abs(y_reg - pred_reg)))
        d_raw = runner.diagnostic_metrics(y_reg, pred_reg, 0, 23.0, 20.0)
        d_adj = runner.diagnostic_metrics(y_reg, adj_reg, 0, 20.0, 20.0)
        self.assertAlmostEqual(d_raw.cal_slope, d_adj.cal_slope, places=12)

    def test_m4_paired_eval_bootstrap_stratification_and_zero_diff(self) -> None:
        rng = np.random.default_rng(2026)
        y_bin = np.array([0] * 40 + [1] * 10)
        idx = runner.eval_bootstrap_indices(y_bin, "F05", k=2, n_boot=50)
        self.assertEqual(idx.shape, (50, 50))
        # Every stratified draw preserves exact class counts (40 zeros, 10 ones)
        self.assertTrue(np.all(y_bin[idx].sum(axis=1) == 10))
        p1 = np.clip(0.15 + 0.35 * y_bin + rng.normal(0.0, 0.1, size=50), 0.02, 0.98)
        pred = np.column_stack((1.0 - p1, p1))
        zero_b = runner.paired_eval_bootstrap_release(y_bin, pred, pred, pred, pred, k=2, idx=idx)
        for val in zero_b.values():
            if np.isfinite(val):
                self.assertLess(abs(val), 1e-12)

        y_reg = rng.normal(25.0, 3.0, size=60)
        idx_r = runner.eval_bootstrap_indices(y_reg, "T2", k=0, n_boot=50)
        pred_r = y_reg + rng.normal(0.0, 1.5, size=60)
        zero_r = runner.paired_eval_bootstrap_release(y_reg, pred_r, pred_r, pred_r, pred_r, k=0, idx=idx_r)
        for val in zero_r.values():
            if np.isfinite(val):
                self.assertLess(abs(val), 1e-12)

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
                    {"task": "T2", "version": "20230612", "training_n": 100, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 2.10, "auroc": np.nan, "r2": 0.20, "rmse": 2.70, "auprc": np.nan, "auprc_lift": np.nan, "log_loss": np.nan, "citl": np.nan, "macro_abs_citl": np.nan, "eo_ratio": np.nan, "ece": np.nan, "cal_me": -0.20, "cal_slope": 0.85, "medae": 1.60, "nmae": 0.68, "prior_adj_error": 2.08, "prior_adj_log_loss": np.nan, "prior_adj_r2": 0.21, "prior_adj_rmse": 2.68},
                    {"task": "T2", "version": "20250321", "training_n": 150, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 2.02, "auroc": np.nan, "r2": 0.23, "rmse": 2.65, "auprc": np.nan, "auprc_lift": np.nan, "log_loss": np.nan, "citl": np.nan, "macro_abs_citl": np.nan, "eo_ratio": np.nan, "ece": np.nan, "cal_me": 0.05, "cal_slope": 0.88, "medae": 1.54, "nmae": 0.65, "prior_adj_error": 2.01, "prior_adj_log_loss": np.nan, "prior_adj_r2": 0.24, "prior_adj_rmse": 2.64},
                    {"task": "T2", "version": "20260810", "training_n": 200, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 1.95, "auroc": np.nan, "r2": 0.26, "rmse": 2.58, "auprc": np.nan, "auprc_lift": np.nan, "log_loss": np.nan, "citl": np.nan, "macro_abs_citl": np.nan, "eo_ratio": np.nan, "ece": np.nan, "cal_me": 0.15, "cal_slope": 0.90, "medae": 1.48, "nmae": 0.63, "prior_adj_error": 1.94, "prior_adj_log_loss": np.nan, "prior_adj_r2": 0.27, "prior_adj_rmse": 2.57},
                    {"task": "F02", "version": "20230612", "training_n": 100, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 1.40, "auroc": np.nan, "r2": 0.16, "rmse": 1.96, "auprc": np.nan, "auprc_lift": np.nan, "log_loss": np.nan, "citl": np.nan, "macro_abs_citl": np.nan, "eo_ratio": np.nan, "ece": np.nan, "cal_me": -0.05, "cal_slope": 0.84, "medae": 0.90, "nmae": 0.65, "prior_adj_error": 1.32, "prior_adj_log_loss": np.nan, "prior_adj_r2": 0.17, "prior_adj_rmse": 1.94},
                    {"task": "F02", "version": "20250321", "training_n": 150, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 1.45, "auroc": np.nan, "r2": 0.15, "rmse": 1.98, "auprc": np.nan, "auprc_lift": np.nan, "log_loss": np.nan, "citl": np.nan, "macro_abs_citl": np.nan, "eo_ratio": np.nan, "ece": np.nan, "cal_me": 0.15, "cal_slope": 0.85, "medae": 1.05, "nmae": 0.68, "prior_adj_error": 1.38, "prior_adj_log_loss": np.nan, "prior_adj_r2": 0.16, "prior_adj_rmse": 1.96},
                    {"task": "F02", "version": "20260810", "training_n": 200, "evaluation_n": 50, "primary_metric": "mae", "observed_release_error": 1.50, "auroc": np.nan, "r2": 0.13, "rmse": 2.00, "auprc": np.nan, "auprc_lift": np.nan, "log_loss": np.nan, "citl": np.nan, "macro_abs_citl": np.nan, "eo_ratio": np.nan, "ece": np.nan, "cal_me": 0.35, "cal_slope": 0.86, "medae": 1.25, "nmae": 0.72, "prior_adj_error": 1.42, "prior_adj_log_loss": np.nan, "prior_adj_r2": 0.15, "prior_adj_rmse": 1.97},
                    {"task": "C05", "version": "20240129", "training_n": 100, "evaluation_n": 50, "primary_metric": "brier", "observed_release_error": 0.050, "auroc": 0.57, "r2": np.nan, "rmse": np.nan, "auprc": 0.09, "auprc_lift": 0.04, "log_loss": 0.20, "citl": -0.05, "macro_abs_citl": 0.05, "eo_ratio": 1.04, "ece": 0.03, "cal_me": np.nan, "cal_slope": 0.28, "medae": np.nan, "nmae": np.nan, "prior_adj_error": 0.049, "prior_adj_log_loss": 0.199, "prior_adj_r2": np.nan, "prior_adj_rmse": np.nan},
                    {"task": "C05", "version": "20250321", "training_n": 150, "evaluation_n": 50, "primary_metric": "brier", "observed_release_error": 0.049, "auroc": 0.58, "r2": np.nan, "rmse": np.nan, "auprc": 0.09, "auprc_lift": 0.04, "log_loss": 0.199, "citl": 0.05, "macro_abs_citl": 0.05, "eo_ratio": 0.96, "ece": 0.028, "cal_me": np.nan, "cal_slope": 0.27, "medae": np.nan, "nmae": np.nan, "prior_adj_error": 0.0485, "prior_adj_log_loss": 0.198, "prior_adj_r2": np.nan, "prior_adj_rmse": np.nan},
                    {"task": "C05", "version": "20260810", "training_n": 200, "evaluation_n": 50, "primary_metric": "brier", "observed_release_error": 0.048, "auroc": 0.57, "r2": np.nan, "rmse": np.nan, "auprc": 0.086, "auprc_lift": 0.039, "log_loss": 0.200, "citl": 0.20, "macro_abs_citl": 0.20, "eo_ratio": 0.82, "ece": 0.026, "cal_me": np.nan, "cal_slope": 0.25, "medae": np.nan, "nmae": np.nan, "prior_adj_error": 0.048, "prior_adj_log_loss": 0.199, "prior_adj_r2": np.nan, "prior_adj_rmse": np.nan},
                ]
            )
            size_df = pd.DataFrame(
                [
                    {"task": "T2", "primary_metric": "mae", "evaluation_n": 50, "training_n": 50, "verified_cut": "20230612", "observed_error": 2.20, "error_p025": 2.00, "error_p975": 2.40, "valid_rounds": 1000, "boundary_group_split": True, "status": "fit", "auroc_mean": np.nan, "auroc_p025": np.nan, "auroc_p975": np.nan, "r2_mean": 0.14, "r2_p025": -0.10, "r2_p975": 0.25, "rmse_mean": 2.80, "rmse_p025": 2.60, "rmse_p975": 3.10, "prior_adj_error_mean": 2.18},
                    {"task": "T2", "primary_metric": "mae", "evaluation_n": 50, "training_n": 100, "verified_cut": "20230612", "observed_error": 2.05, "error_p025": 1.98, "error_p975": 2.12, "valid_rounds": 1000, "boundary_group_split": True, "status": "fit", "auroc_mean": np.nan, "auroc_p025": np.nan, "auroc_p975": np.nan, "r2_mean": 0.21, "r2_p025": 0.12, "r2_p975": 0.26, "rmse_mean": 2.68, "rmse_p025": 2.58, "rmse_p975": 2.79, "prior_adj_error_mean": 2.04},
                    {"task": "T2", "primary_metric": "mae", "evaluation_n": 50, "training_n": 200, "verified_cut": "20260810", "observed_error": 1.95, "error_p025": 1.95, "error_p975": 1.95, "valid_rounds": 1000, "boundary_group_split": False, "status": "fit", "auroc_mean": np.nan, "auroc_p025": np.nan, "auroc_p975": np.nan, "r2_mean": 0.26, "r2_p025": 0.26, "r2_p975": 0.26, "rmse_mean": 2.58, "rmse_p025": 2.58, "rmse_p975": 2.58, "prior_adj_error_mean": 1.94},
                    {"task": "F02", "primary_metric": "mae", "evaluation_n": 50, "training_n": 100, "verified_cut": "20230612", "observed_error": 1.45, "error_p025": 1.35, "error_p975": 1.55, "valid_rounds": 1000, "boundary_group_split": True, "status": "fit", "auroc_mean": np.nan, "auroc_p025": np.nan, "auroc_p975": np.nan, "r2_mean": 0.04, "r2_p025": -0.20, "r2_p975": 0.15, "rmse_mean": 2.10, "rmse_p025": 1.96, "rmse_p975": 2.30, "prior_adj_error_mean": 1.38},
                    {"task": "F02", "primary_metric": "mae", "evaluation_n": 50, "training_n": 200, "verified_cut": "20260810", "observed_error": 1.50, "error_p025": 1.50, "error_p975": 1.50, "valid_rounds": 1000, "boundary_group_split": False, "status": "fit", "auroc_mean": np.nan, "auroc_p025": np.nan, "auroc_p975": np.nan, "r2_mean": 0.13, "r2_p025": 0.13, "r2_p975": 0.13, "rmse_mean": 2.00, "rmse_p025": 2.00, "rmse_p975": 2.00, "prior_adj_error_mean": 1.42},
                    {"task": "C05", "primary_metric": "brier", "evaluation_n": 50, "training_n": 100, "verified_cut": "20230612", "observed_error": np.nan, "error_p025": np.nan, "error_p975": np.nan, "valid_rounds": 0, "boundary_group_split": True, "status": "missing_training_class", "auroc_mean": np.nan, "auroc_p025": np.nan, "auroc_p975": np.nan, "r2_mean": np.nan, "r2_p025": np.nan, "r2_p975": np.nan, "rmse_mean": np.nan, "rmse_p025": np.nan, "rmse_p975": np.nan, "prior_adj_error_mean": np.nan},
                    {"task": "C05", "primary_metric": "brier", "evaluation_n": 50, "training_n": 200, "verified_cut": "20260810", "observed_error": 0.048, "error_p025": 0.048, "error_p975": 0.048, "valid_rounds": 1000, "boundary_group_split": False, "status": "fit", "auroc_mean": 0.57, "auroc_p025": 0.57, "auroc_p975": 0.57, "r2_mean": np.nan, "r2_p025": np.nan, "r2_p975": np.nan, "rmse_mean": np.nan, "rmse_p025": np.nan, "rmse_p975": np.nan, "prior_adj_error_mean": 0.048},
                ]
            )
            paired_df = pd.DataFrame(
                [
                    {"task": "T2", "first_version": "20230612", "latest_version": "20260810", "rel_err_diff": -0.15, "rel_err_ci_lo": -0.25, "rel_err_ci_hi": -0.05, "rel_adj_err_diff": -0.14, "rel_adj_err_ci_lo": -0.24, "rel_adj_err_ci_hi": -0.04, "rel_last_raw_minus_adj": 0.01, "rel_last_raw_minus_adj_ci_lo": -0.02, "rel_last_raw_minus_adj_ci_hi": 0.04, "step_err_diff": -0.10, "step_err_ci_lo": -0.18, "step_err_ci_hi": -0.02, "step_adj_err_diff": -0.10, "step_adj_err_ci_lo": -0.18, "step_adj_err_ci_hi": -0.02, "span_err_diff": -0.25, "span_err_ci_lo": -0.35, "span_err_ci_hi": -0.15},
                    {"task": "F02", "first_version": "20230612", "latest_version": "20260810", "rel_err_diff": 0.10, "rel_err_ci_lo": 0.03, "rel_err_ci_hi": 0.17, "rel_adj_err_diff": 0.10, "rel_adj_err_ci_lo": 0.03, "rel_adj_err_ci_hi": 0.17, "rel_last_raw_minus_adj": 0.08, "rel_last_raw_minus_adj_ci_lo": 0.04, "rel_last_raw_minus_adj_ci_hi": 0.12, "step_err_diff": 0.05, "step_err_ci_lo": 0.01, "step_err_ci_hi": 0.09, "step_adj_err_diff": 0.04, "step_adj_err_ci_lo": 0.00, "step_adj_err_ci_hi": 0.08, "span_err_diff": 0.05, "span_err_ci_lo": -0.01, "span_err_ci_hi": 0.11},
                    {"task": "C05", "first_version": "20240129", "latest_version": "20260810", "rel_err_diff": -0.002, "rel_err_ci_lo": -0.006, "rel_err_ci_hi": 0.002, "rel_adj_err_diff": -0.001, "rel_adj_err_ci_lo": -0.005, "rel_adj_err_ci_hi": 0.003, "rel_last_raw_minus_adj": 0.000, "rel_last_raw_minus_adj_ci_lo": -0.001, "rel_last_raw_minus_adj_ci_hi": 0.001, "step_err_diff": np.nan, "step_err_ci_lo": np.nan, "step_err_ci_hi": np.nan, "step_adj_err_diff": np.nan, "step_adj_err_ci_lo": np.nan, "step_adj_err_ci_hi": np.nan, "span_err_diff": np.nan, "span_err_ci_lo": np.nan, "span_err_ci_hi": np.nan},
                ]
            )
            tasks_df.to_csv(res_dir / "task_index.csv", index=False)
            versions_df.to_csv(res_dir / "real_version_trajectory.csv", index=False)
            size_df.to_csv(res_dir / "chronological_size_trajectory.csv", index=False)
            paired_df.to_csv(res_dir / "paired_uncertainty_and_recalibration.csv", index=False)

            t, v, s = assets.load_inputs(res_dir)
            p = assets.load_paired(res_dir)
            desc = assets.describe_trajectories(t, v, s)
            self.assertEqual(desc.loc[desc.task == "T2", "direction"].iloc[0], "Declining")
            self.assertEqual(desc.loc[desc.task == "C05", "direction"].iloc[0], "Single tier")
            assets.make_tables(desc, s, res_dir, paired=p)
            assets.make_task_catalog(t, res_dir)
            assets.make_figures(t, v, s, desc, res_dir, fig_dir, ["T2", "F02", "C05"])
            self.assertTrue((res_dir / "paper_design_table.tex").is_file())
            self.assertTrue((res_dir / "paper_family_table.tex").is_file())
            self.assertTrue((res_dir / "paper_task_catalog.tex").is_file())
            self.assertTrue((res_dir / "paper_task_results.tex").is_file())
            self.assertTrue((res_dir / "paper_secondary_classification.tex").is_file())
            self.assertTrue((res_dir / "paper_calibration_classification.tex").is_file())
            self.assertTrue((res_dir / "paper_secondary_regression.tex").is_file())
            self.assertTrue((res_dir / "paper_paired_bootstrap.tex").is_file())
            self.assertTrue((fig_dir / "chronological_examples.pdf").is_file())
            self.assertTrue((res_dir / "supp_fig_s1.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
