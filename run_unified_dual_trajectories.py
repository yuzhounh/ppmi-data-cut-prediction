"""End-to-end unified runner for P12 official-release and chronological-size trajectories."""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import ppmi_task_registry as reg

AGE_TOLERANCE = 0.01  # Years (~3.65 days): absorbs CSV 4-decimal rounding and 365/365.25 age divisor shifts.
SEED = 20261121
ROUNDS = 1000
PILOT_ROUNDS = 10
REFERENCE_TOLERANCE = 1e-7

_IN_CODE_DIR = Path(__file__).resolve().parent.name == "code"
DEFAULT_RESULTS = (
    reg.PROJECT / "manuscript" / "results" / "unified_dual_trajectories"
    if _IN_CODE_DIR
    else reg.PROJECT / "outputs" / "results"
)
DEFAULT_CACHE = reg.PROJECT / "data" / "derived" / "unified_dual_trajectories"


@dataclass(frozen=True)
class TrainingRecord:
    baseline_month: int
    outcome_month: int
    x: tuple
    y: float | int | str


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_arrays(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def halving(total: int) -> tuple[int, ...]:
    end = min(range(total.bit_length()), key=lambda k: abs(math.log(total / (30 * 2**k))))
    return tuple(math.floor(total / 2**k + 0.5) for k in range(end + 1))


def random_stream(task_id: str, repeat: int, purpose: int = 2) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([SEED, int.from_bytes(task_id.encode("ascii")), repeat, purpose])
    )


def records_match(observed: TrainingRecord | None, reference: TrainingRecord, age_index: int = 0) -> bool:
    """Compare an archived record against the reference record with tolerance on continuous age only."""
    if observed is None:
        return False
    if (
        observed.baseline_month != reference.baseline_month
        or observed.outcome_month != reference.outcome_month
        or observed.y != reference.y
    ):
        return False
    if len(observed.x) != len(reference.x):
        return False
    for idx, (a, b) in enumerate(zip(observed.x, reference.x)):
        if (a is None) != (b is None):
            return False
        if a is None:
            continue
        if idx == age_index:
            if abs(a - b) > AGE_TOLERANCE:
                return False
        elif a != b:
            return False
    return True


def candidate_records(task: reg.Task, eligible_records: dict, people: dict) -> dict[str, TrainingRecord]:
    rows = {}
    for pid, (label_val, _) in eligible_records.items():
        _, baseline, followup, baseline_month, _ = people[pid]
        outcome = baseline_month if task.mode == "BL" else followup.get("visit_date")
        feature_row = followup if task.mode == "VISIT24" else baseline
        assert outcome is not None
        rows[pid] = TrainingRecord(
            baseline_month,
            int(outcome),
            tuple(feature_row.get(field) for field in task.features),
            label_val,
        )
    return rows


def arrays_from(records: dict[str, TrainingRecord], ids: list[str], labels: tuple[str, ...]):
    classes = {value: i for i, value in enumerate(labels)}
    x = np.asarray([records[p].x for p in ids], dtype=float)
    y = np.asarray([classes[records[p].y] if classes else records[p].y for p in ids], dtype=int if classes else float)
    assert np.isfinite(y).all() and np.isfinite(x).any(axis=1).all()
    return x, y


def preprocess(train_x: np.ndarray, eval_x: np.ndarray):
    """Median imputation and missingness indicators fitted on train_x only, followed by standardization."""
    missing = np.isnan(train_x)
    eval_missing = np.isnan(eval_x)
    indicators = missing.any(axis=0)
    required = indicators | eval_missing.any(axis=0)
    if required.any():
        medians = np.zeros(train_x.shape[1])
        for col in np.flatnonzero(required):
            present = train_x[~missing[:, col], col]
            if present.size:
                medians[col] = np.median(present)
        train_x = np.where(missing, medians, train_x)
        eval_x = np.where(eval_missing, medians, eval_x)
    if indicators.any():
        train_x = np.column_stack((train_x, missing[:, indicators]))
        eval_x = np.column_stack((eval_x, eval_missing[:, indicators]))
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    return (train_x - mean) / scale, (eval_x - mean) / scale


def make_classifier() -> LogisticRegression:
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(solver="lbfgs", C=1.0, l1_ratio=0.0, max_iter=2000)
    return LogisticRegression(solver="lbfgs", C=1.0, penalty="l2", max_iter=2000)


def predict(train_x: np.ndarray, y: np.ndarray, eval_x: np.ndarray, num_classes: int) -> np.ndarray:
    x_mat, e_mat = preprocess(train_x, eval_x)
    if num_classes:
        return make_classifier().fit(x_mat, y).predict_proba(e_mat)
    x_mean = x_mat.mean(axis=0)
    x_centered = x_mat - x_mean
    y_mean = y.mean()
    gram = x_centered.T @ x_centered
    gram.flat[:: len(gram) + 1] += 1.0
    coef = np.linalg.solve(gram, x_centered.T @ (y - y_mean))
    return (e_mat - x_mean) @ coef + y_mean


def reference_predict(train_x: np.ndarray, y: np.ndarray, eval_x: np.ndarray, num_classes: int) -> np.ndarray:
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        make_classifier() if num_classes else Ridge(alpha=1.0),
    )
    pipeline.fit(train_x, y)
    return pipeline.predict_proba(eval_x) if num_classes else pipeline.predict(eval_x)


def case_loss(y: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes == 2:
        return (y - pred[:, 1]) ** 2
    if num_classes > 2:
        squared = (pred**2).sum(axis=1)
        return squared - 2.0 * pred[np.arange(len(y)), y] + 1.0
    return np.abs(y - pred)


def label_counts(values: np.ndarray, nclasses: int) -> np.ndarray:
    return np.bincount(values, minlength=nclasses) if nclasses else np.empty(0, dtype=int)


def run_identity(data_dir: Path) -> dict:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(reg.__file__)):
        digest.update(path.read_bytes())
    snapshots = {}
    for version in reg.VERSIONS:
        info = (data_dir / reg.extract_filename(version)).stat()
        snapshots[version] = [info.st_size, info.st_mtime_ns]
    return {
        "code_digest": digest.hexdigest(),
        "snapshots": snapshots,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "reserve_seed": reg.RESERVE_SEED,
        "reserve_fraction": reg.RESERVE_FRACTION,
        "min_releases": reg.MIN_RELEASES,
        "tie_seed": SEED,
        "tie_rounds": ROUNDS,
        "age_tolerance_years": AGE_TOLERANCE,
    }


def missingness_summary(x: np.ndarray, features: tuple[str, ...]) -> dict[str, float]:
    return {feat: round(float(np.isnan(x[:, idx]).mean() * 100.0), 2) for idx, feat in enumerate(features)}


def target_summary(y: np.ndarray, labels: tuple[str, ...]) -> dict:
    if labels:
        cnt = np.bincount(y, minlength=len(labels))
        total = int(cnt.sum())
        return {
            "n": total,
            "counts": {lbl: int(cnt[i]) for i, lbl in enumerate(labels)},
            "shares_pct": {lbl: round(100.0 * int(cnt[i]) / total, 2) if total else 0.0 for i, lbl in enumerate(labels)},
        }
    return {
        "n": int(len(y)),
        "mean": round(float(np.mean(y)), 4),
        "sd": round(float(np.std(y, ddof=0)), 4),
        "median": round(float(np.median(y)), 4),
    }


def prepare_unified_dataset(data_dir: Path, results_dir: Path, cache_dir: Path, identity: dict):
    inputs_dir = cache_dir / "inputs"
    marker = inputs_dir / "manifest.json"
    if (
        marker.exists()
        and (results_dir / "task_index.csv").exists()
        and (results_dir / "inventory_summary.json").exists()
        and read_json(marker)["identity"] == identity
    ):
        return pd.read_csv(results_dir / "task_index.csv", dtype={"first_version": str, "latest_version": str}), read_json(
            results_dir / "inventory_summary.json"
        )

    print("Loading all 11 official curated extracts under unified rules...", flush=True)
    versions = reg.VERSIONS
    all_columns = set.union(*(t.columns for t in reg.ALL_CANDIDATE_TASKS), {"PATNO", "EVENT_ID", "visit_date", "COHORT"})
    task_data = {t.id: {} for t in reg.ALL_CANDIDATE_TASKS}
    people_by_cut = {}
    cohorts = {}
    headers_by_cut = {}

    for version in versions:
        headers, people, cutoff = reg.people_for(data_dir, version, all_columns)
        people_by_cut[version] = people
        cohorts[version] = {pid: row[0] for pid, row in people.items()}
        headers_by_cut[version] = sorted(headers & all_columns)
        for task in reg.ALL_CANDIDATE_TASKS:
            task_data[task.id][version] = reg.collect(task, headers, people, cutoff)
        print(f"  {version}: {len(people):,} baseline participants", flush=True)

    reserved, reserve_meta = reg.unified_reserve(versions, cohorts)
    aliases = {}
    if all(task_data["F05"][v] == task_data["F07"][v] for v in versions):
        aliases["F07"] = "F05"

    report = {}
    support_rows, status_rows = [], []
    admitted_specs: list[tuple[reg.Task, tuple[str, ...], dict]] = []

    for task in reg.ALL_CANDIDATE_TASKS:
        observations = task_data[task.id]
        comparisons = [
            (versions[i:], reg.comparison(task, versions[i:], observations, reserved))
            for i in range(len(versions) - reg.MIN_RELEASES + 1)
        ]
        eleven = comparisons[0][1]
        candidates = [(cuts, stats) for cuts, stats in comparisons if reg.supported(task, stats)]
        if task.id in aliases:
            decision = "existing_equivalent"
        elif task.qualification:
            decision = "conditional_or_not_in_plan"
        elif candidates:
            decision = "main_11" if len(candidates[0][0]) == len(versions) else "limited_release_series"
        else:
            decision = "insufficient_support"

        proposed_cuts, proposed = candidates[0] if candidates else next(
            ((cuts, stats) for cuts, stats in comparisons if stats["dev"]), ((), eleven)
        )
        if decision == "conditional_or_not_in_plan":
            proposed_cuts, proposed = next(((cuts, stats) for cuts, stats in comparisons if stats["dev"]), ((), eleven))

        if decision in {"main_11", "limited_release_series"}:
            admitted_specs.append((task, proposed_cuts, proposed))

        report[task.id] = {
            "title": task.title,
            "family": task.family,
            "definition": task.__dict__,
            "decision": decision,
            "all_eleven": eleven,
            "proposed_cuts": list(proposed_cuts),
            "proposed": proposed,
            "equivalent_to": aliases.get(task.id),
        }
        status_rows.append(
            {
                "task": task.id,
                "family": task.family,
                "title": task.title,
                "decision": decision,
                "proposed_versions": len(proposed_cuts),
                "first_proposed_version": proposed_cuts[0] if proposed_cuts else "",
                "common_ref_all_11": eleven["common_ref_n"],
                "strict_common_all_11": eleven["strict_common_n"],
                "fixed_eval_all_11": eleven["eval_n"],
                "classes_fixed_eval_all_11": json.dumps(eleven["eval_classes"]),
                "common_ref_proposed": proposed["common_ref_n"],
                "strict_common_proposed": proposed["strict_common_n"],
                "fixed_eval_proposed": proposed["eval_n"],
                "classes_fixed_eval_proposed": json.dumps(proposed["eval_classes"]),
                "min_dev_n_proposed": proposed["min_dev_n"],
                "min_dev_class_proposed": proposed["min_dev_class"],
                "qualification": task.qualification,
                "equivalent_to": aliases.get(task.id),
            }
        )
        for version, (availability, base, valid, precursor) in observations.items():
            dev_ids = set(valid) - reserved
            support_rows.append(
                {
                    "task": task.id,
                    "family": task.family,
                    "title": task.title,
                    "version": version,
                    "availability": availability,
                    "baseline_eligible": len(base) if availability == "available" else None,
                    "effective_n": len(valid) if availability == "available" else None,
                    "all_inputs_complete": sum(v[1] for v in valid.values()) if availability == "available" else None,
                    "class_counts": json.dumps(reg.counts(valid, valid, task)) if availability == "available" else None,
                    "development_n": len(dev_ids) if availability == "available" else None,
                    "development_class_counts": (
                        json.dumps(reg.counts(valid, dev_ids, task)) if availability == "available" else None
                    ),
                    "common_eval_all_11_n": eleven["eval_n"],
                    "common_eval_proposed_n": proposed["eval_n"] if version in proposed_cuts else None,
                    "precursor_complete_n": precursor,
                }
            )

    # Build cached per-task release arrays and chronological backward-verification arrays
    inputs_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for task, cuts, proposed in admitted_specs:
        assert task.features[0] == "age", f"Expected age at feature index 0 for {task.id}"
        assert "COHORT" not in task.features and (task.mode == "M24" or task.target not in task.features)
        num_classes = len(task.labels)
        metric = {0: "mae", 2: "brier", 3: "multiclass_brier"}[num_classes]

        release_arrays = {}
        common_eligible = None
        by_version_dev: dict[str, dict[str, TrainingRecord]] = {}
        ref_all_records: dict[str, TrainingRecord] | None = None

        for version in versions:
            availability, base_ids, valid, _ = task_data[task.id][version]
            extracted = candidate_records(task, valid, people_by_cut[version]) if availability == "available" else {}
            dev_records = {pid: rec for pid, rec in extracted.items() if pid not in reserved}
            by_version_dev[version] = dev_records
            if version in cuts:
                assert availability == "available"
                common_eligible = base_ids.copy() if common_eligible is None else (common_eligible & base_ids)
                dev_ids = sorted(dev_records)
                expected_n, expected_cls = proposed["dev"][version]
                assert len(dev_ids) == expected_n and reg.counts(valid, dev_ids, task) == expected_cls
                assert not (set(dev_ids) & reserved)
                x_cut, y_cut = arrays_from(dev_records, dev_ids, task.labels)
                release_arrays[f"x_{version}"] = x_cut
                release_arrays[f"y_{version}"] = y_cut
            if version == reg.REFERENCE:
                ref_all_records = extracted

        assert ref_all_records is not None and common_eligible is not None
        eval_ids = sorted(set(ref_all_records) & common_eligible & reserved)
        assert len(eval_ids) == proposed["eval_n"]
        eval_x, eval_y = arrays_from(ref_all_records, eval_ids, task.labels)
        ref_dev_records = by_version_dev[reg.REFERENCE]
        ref_dev_ids = sorted(ref_dev_records)
        assert set(ref_dev_ids).isdisjoint(eval_ids)

        # Trace each August 2026 development record backwards across all 11 archived extracts
        membership = np.zeros((len(versions), len(ref_dev_ids)), dtype=bool)
        is_baseline = task.mode == "BL"
        for j, version in enumerate(versions):
            observed = by_version_dev[version]
            cut_month = reg.snapshot_month(version)
            for i, pid in enumerate(ref_dev_ids):
                ref_rec = ref_dev_records[pid]
                if records_match(observed.get(pid), ref_rec, age_index=0):
                    assert ref_rec.baseline_month + (0 if is_baseline else 27) <= cut_month
                    assert ref_rec.outcome_month <= cut_month
                    membership[j, i] = True

        assert membership[-1].all()
        stable = np.logical_and.accumulate(membership[::-1], axis=0)
        first_idx = len(versions) - stable.sum(axis=0)
        order = np.lexsort((np.arange(len(ref_dev_ids)), first_idx))
        chrono_x, chrono_y = arrays_from(ref_dev_records, ref_dev_ids, task.labels)

        save_arrays(
            inputs_dir / f"{task.id}.npz",
            **release_arrays,
            eval_x=eval_x,
            eval_y=eval_y,
            eval_ids=np.asarray(eval_ids, dtype=str),
            chrono_x=chrono_x[order],
            chrono_y=chrono_y[order],
            chrono_ids=np.asarray(ref_dev_ids, dtype=str)[order],
            cutoff_idx=first_idx[order],
            membership=membership[:, order],
            baseline_month=np.asarray([ref_dev_records[p].baseline_month for p in ref_dev_ids], dtype=int)[order],
            outcome_month=np.asarray([ref_dev_records[p].outcome_month for p in ref_dev_ids], dtype=int)[order],
            classes=np.asarray(task.labels, dtype=str),
            cuts=np.asarray(cuts, dtype=str),
        )

        index_rows.append(
            {
                "task": task.id,
                "family": reg.FAMILY_LABELS[task.family],
                "title": task.title,
                "timing": task.mode,
                "versions": len(cuts),
                "first_version": cuts[0],
                "latest_version": cuts[-1],
                "evaluation_n": len(eval_ids),
                "evaluation_classes": json.dumps(proposed["eval_classes"]),
                "primary_metric": metric,
                "features": json.dumps(list(task.features)),
                "dev_missingness_pct": json.dumps(missingness_summary(chrono_x, task.features)),
                "eval_missingness_pct": json.dumps(missingness_summary(eval_x, task.features)),
                "first_dev_target_summary": json.dumps(target_summary(release_arrays[f"y_{cuts[0]}"], task.labels)),
                "latest_dev_target_summary": json.dumps(target_summary(release_arrays[f"y_{cuts[-1]}"], task.labels)),
                "eval_target_summary": json.dumps(target_summary(eval_y, task.labels)),
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "support_by_release.csv", support_rows)
    write_csv(results_dir / "common_evaluation_and_admission.csv", status_rows)
    write_csv(results_dir / "task_index.csv", index_rows)
    write_csv(cache_dir / "reserved_evaluation_identities_restricted.csv", [{"participant": p} for p in sorted(reserved)])

    summary = {
        "versions": list(versions),
        "reference": reg.REFERENCE,
        "min_releases": reg.MIN_RELEASES,
        "unified_partition": reserve_meta,
        "tasks": report,
        "relevant_headers": headers_by_cut,
        "counts": dict(Counter(r["decision"] for r in status_rows)),
        "equivalences": aliases,
    }
    save_json(results_dir / "inventory_summary.json", summary)
    save_json(marker, {"identity": identity, "admitted_tasks": len(admitted_specs), "archive_cuts": list(versions)})
    print(
        f"Prepared unified dataset: {len(admitted_specs)} admitted tasks, "
        f"{sum(len(c) for _, c, _ in admitted_specs)} official release points, "
        f"{len(reserved):,} reserved evaluation identities.",
        flush=True,
    )
    return pd.read_csv(results_dir / "task_index.csv", dtype={"first_version": str, "latest_version": str}), summary


def checked_stages(block, versions: tuple[str, ...], timing: str):
    x, y = block["chrono_x"], block["chrono_y"]
    ids, evaluation = block["chrono_ids"], block["eval_ids"]
    cut_rank, clinical_month = block["cutoff_idx"], block["outcome_month"]
    baseline = block["baseline_month"]
    membership = block["membership"]
    full = len(y)
    assert len(np.unique(ids)) == full
    assert not (set(ids) & set(evaluation))
    assert np.isfinite(y).all() and np.all(np.diff(cut_rank) >= 0)
    assert membership[-1].all()
    former_n = 0
    last_cut = -1
    stages = []
    for n in reversed(halving(full)):
        cutoff = int(cut_rank[n - 1])
        cutoff_month = reg.snapshot_month(versions[cutoff])
        assert cutoff >= last_cut and np.all(cut_rank[former_n:n] >= last_cut)
        assert n > former_n
        assert membership[cutoff, :n].all()
        assert (baseline[:n] + (27 if timing == "M24" else 0) <= cutoff_month).all()
        assert (clinical_month[:n] <= cutoff_month).all()
        assert x[:n].shape[0] == n and not np.isinf(x[:n]).any()
        earlier_n = int(np.count_nonzero(cut_rank < cutoff))
        tied_total = int(np.count_nonzero(cut_rank == cutoff))
        tied_used = n - earlier_n
        assert 1 <= tied_used <= tied_total
        stages.append(
            {
                "training_n": n,
                "verified_cut": versions[cutoff],
                "cutoff_idx": cutoff,
                "observation_through": reg.calendar_month(cutoff_month),
                "earlier_fixed_n": earlier_n,
                "boundary_group_n": tied_total,
                "used_from_boundary_group": tied_used,
                "boundary_group_split": tied_used < tied_total,
            }
        )
        former_n, last_cut = n, cutoff
    return stages


def tie_permutation(cut_rank: np.ndarray, task_id: str, repeat: int) -> np.ndarray:
    """Permute only within each verified snapshot group, preserving strict snapshot ordering."""
    rng = random_stream(task_id, repeat, purpose=2)
    order = np.arange(len(cut_rank))
    starts = np.flatnonzero(np.concatenate(([True], np.diff(cut_rank) > 0)))
    ends = np.concatenate((starts[1:], [len(cut_rank)]))
    for s, e in zip(starts, ends):
        if e - s > 1:
            order[s:e] = rng.permutation(order[s:e])
    return order


def evaluate_task(task_row: dict, cache_dir: Path, rounds: int, identity: dict, pilot: bool = False) -> dict:
    key = task_row["task"]
    timing = task_row["timing"]
    primary_metric = task_row["primary_metric"]
    evaluation_n = int(task_row["evaluation_n"])
    versions = reg.VERSIONS

    with np.load(cache_dir / "inputs" / f"{key}.npz") as block:
        cuts = [str(c) for c in block["cuts"]]
        eval_x, eval_y = block["eval_x"], block["eval_y"]
        classes = [str(c) for c in block["classes"]]
        k = len(classes)
        max_ref_diff = 0.0

        # Trajectory 1: official curated releases
        release_rows = []
        started = time.perf_counter()
        for cut in cuts:
            x_cut, y_cut = block[f"x_{cut}"], block[f"y_{cut}"]
            pred = predict(x_cut, y_cut, eval_x, k)
            loss = case_loss(eval_y, pred, k)
            assert np.isfinite(loss).all()
            if pilot:
                direct = reference_predict(x_cut, y_cut, eval_x, k)
                max_ref_diff = max(max_ref_diff, float(np.max(np.abs(direct - pred))))
            cls_counts = {classes[c]: int(np.count_nonzero(y_cut == c)) for c in range(k)} if k else {}
            release_rows.append(
                {
                    "task": key,
                    "version": cut,
                    "training_n": int(len(y_cut)),
                    "evaluation_n": evaluation_n,
                    "primary_metric": primary_metric,
                    "observed_release_error": float(loss.mean()),
                    "development_class_counts": json.dumps(cls_counts),
                }
            )
        expected_anchor = release_rows[-1]["observed_release_error"]

        # Trajectory 2: chronological doubling within August 2026 reference
        stages = checked_stages(block, versions, timing)
        x, y = block["chrono_x"], block["chrono_y"]
        cut_rank = block["cutoff_idx"]
        n_stages = len(stages)
        assert stages[-1]["training_n"] == release_rows[-1]["training_n"]

        scores = np.full((rounds, n_stages), np.nan)
        class_totals = np.zeros((rounds, n_stages, k), dtype=int) if k else np.zeros((rounds, n_stages, 0), dtype=int)

        unsplit_cache = {}
        for col, stage in enumerate(stages):
            if not stage["boundary_group_split"]:
                n = stage["training_n"]
                sub_y = y[:n]
                cnt = label_counts(sub_y, k)
                if k and not cnt.all():
                    unsplit_cache[col] = (np.nan, cnt)
                else:
                    pred = predict(x[:n], sub_y, eval_x, k)
                    loss = case_loss(eval_y, pred, k)
                    assert np.isfinite(loss).all()
                    if pilot:
                        direct = reference_predict(x[:n], sub_y, eval_x, k)
                        max_ref_diff = max(max_ref_diff, float(np.max(np.abs(direct - pred))))
                    unsplit_cache[col] = (float(loss.mean()), cnt)

        assert n_stages - 1 in unsplit_cache
        assert np.isclose(unsplit_cache[n_stages - 1][0], expected_anchor, rtol=0, atol=1e-9), (
            key,
            unsplit_cache[n_stages - 1][0],
            expected_anchor,
        )

        ckpt_path = cache_dir / f"{key}_tie_checkpoint.npz"
        start = 0
        if not pilot and ckpt_path.exists():
            with np.load(ckpt_path) as ckpt:
                if ckpt["identity"].item() == identity["code_digest"] and ckpt["scores"].shape[1] == n_stages:
                    start = min(rounds, int(ckpt["completed"].item()))
                    scores[:start] = ckpt["scores"][:start]
                    if k:
                        class_totals[:start] = ckpt["class_totals"][:start]
                    max_ref_diff = max(max_ref_diff, float(ckpt["max_ref_diff"].item()))

        split_cols = [c for c, st in enumerate(stages) if st["boundary_group_split"]]
        check_cols = {split_cols[0], split_cols[-1]} if split_cols else set()

        for r in range(start, rounds):
            order = tie_permutation(cut_rank, key, r)
            if pilot:
                assert np.array_equal(cut_rank[order], cut_rank)
                assert len(np.unique(order)) == len(cut_rank)
                prev_set: set[int] = set()
            for col, stage in enumerate(stages):
                n = stage["training_n"]
                if col in unsplit_cache:
                    val, cnt = unsplit_cache[col]
                    scores[r, col] = val
                    if k:
                        class_totals[r, col] = cnt
                    if pilot:
                        cur_set = set(order[:n])
                        assert prev_set < cur_set
                        prev_set = cur_set
                    continue
                idx = order[:n]
                if pilot:
                    cur_set = set(idx)
                    assert len(cur_set) == n and prev_set < cur_set
                    earlier_fixed = set(np.flatnonzero(cut_rank < stage["cutoff_idx"]))
                    assert earlier_fixed <= cur_set
                    prev_set = cur_set
                sub_y = y[idx]
                cnt = label_counts(sub_y, k)
                if k:
                    class_totals[r, col] = cnt
                if k and not cnt.all():
                    continue
                sub_x = x[idx]
                pred = predict(sub_x, sub_y, eval_x, k)
                loss = case_loss(eval_y, pred, k)
                assert np.isfinite(loss).all()
                if pilot and col in check_cols:
                    direct = reference_predict(sub_x, sub_y, eval_x, k)
                    max_ref_diff = max(max_ref_diff, float(np.max(np.abs(direct - pred))))
                scores[r, col] = float(loss.mean())
            if not pilot and ((r + 1) % 250 == 0 or r + 1 == rounds):
                save_arrays(
                    ckpt_path,
                    identity=identity["code_digest"],
                    completed=r + 1,
                    scores=scores[: r + 1],
                    class_totals=class_totals[: r + 1],
                    max_ref_diff=max_ref_diff,
                )

        elapsed = time.perf_counter() - started
        if pilot:
            assert max_ref_diff < REFERENCE_TOLERANCE, (key, max_ref_diff)
            return {
                "task": key,
                "release_points": len(release_rows),
                "checked_stages": n_stages,
                "split_stages": len(split_cols),
                "first_cut": stages[0]["verified_cut"],
                "last_cut": stages[-1]["verified_cut"],
                "max_reference_prediction_error": max_ref_diff,
                "fit_seconds": elapsed,
                "missing_class_draws": int(np.isnan(scores).sum()),
            }

        stage_rows = []
        for col, stage in enumerate(stages):
            vals = scores[:, col]
            finite = vals[np.isfinite(vals)]
            valid = len(finite)
            missing = rounds - valid
            if valid:
                p025, med, p975 = np.percentile(finite, [2.5, 50.0, 97.5])
                mean_err = float(finite.mean())
                sd_err = float(finite.std(ddof=0))
            else:
                p025 = med = p975 = mean_err = sd_err = np.nan
            if k:
                if stage["boundary_group_split"]:
                    mean_counts = {classes[c]: round(float(class_totals[:, col, c].mean()), 2) for c in range(k)}
                else:
                    mean_counts = {classes[c]: int(class_totals[0, col, c]) for c in range(k)}
            else:
                mean_counts = {}
            status = "fit" if missing == 0 else ("missing_training_class" if valid == 0 else "partial_missing_class")
            stage_rows.append(
                {
                    "task": key,
                    "primary_metric": primary_metric,
                    "evaluation_n": evaluation_n,
                    "training_n": stage["training_n"],
                    "verified_cut": stage["verified_cut"],
                    "observation_through": stage["observation_through"],
                    "earlier_fixed_n": stage["earlier_fixed_n"],
                    "boundary_group_n": stage["boundary_group_n"],
                    "used_from_boundary_group": stage["used_from_boundary_group"],
                    "boundary_group_split": stage["boundary_group_split"],
                    "observed_error": mean_err,
                    "error_mean": mean_err,
                    "error_sd": sd_err,
                    "error_p025": float(p025) if np.isfinite(p025) else np.nan,
                    "error_median": float(med) if np.isfinite(med) else np.nan,
                    "error_p975": float(p975) if np.isfinite(p975) else np.nan,
                    "attempted_rounds": rounds,
                    "valid_rounds": valid,
                    "missing_class_rounds": missing,
                    "class_counts": json.dumps(mean_counts),
                    "status": status,
                }
            )

        step_rows = []
        for col in range(1, n_stages):
            diff = scores[:, col] - scores[:, col - 1]
            finite = diff[np.isfinite(diff)]
            valid = len(finite)
            if valid:
                d025, d50, d975 = np.percentile(finite, [2.5, 50.0, 97.5])
                dmean = float(finite.mean())
                p_dec = float((finite < 0).mean())
            else:
                d025 = d50 = d975 = dmean = p_dec = np.nan
            step_rows.append(
                {
                    "task": key,
                    "primary_metric": primary_metric,
                    "smaller_n": stages[col - 1]["training_n"],
                    "larger_n": stages[col]["training_n"],
                    "smaller_cut": stages[col - 1]["verified_cut"],
                    "larger_cut": stages[col]["verified_cut"],
                    "valid_paired_rounds": valid,
                    "delta_mean": dmean,
                    "delta_p025": float(d025) if np.isfinite(d025) else np.nan,
                    "delta_median": float(d50) if np.isfinite(d50) else np.nan,
                    "delta_p975": float(d975) if np.isfinite(d975) else np.nan,
                    "prob_decreased": p_dec,
                }
            )
        print(f"{key} {rounds}r {elapsed:.2f}s (cuts={len(cuts)}, splits={len(split_cols)})", flush=True)
        return {
            "task": key,
            "releases": release_rows,
            "stages": stage_rows,
            "steps": step_rows,
            "fit_seconds": elapsed,
            "max_reference_prediction_error": max_ref_diff,
        }


def _worker(args):
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        return evaluate_task(*args)


def run_all_tasks(index: pd.DataFrame, cache_dir: Path, rounds: int, identity: dict, workers: int, pilot: bool = False):
    jobs = [(row._asdict(), cache_dir, rounds, identity, pilot) for row in index.itertuples()]
    if workers == 1:
        return [_worker(job) for job in jobs]
    # Submit larger tasks first for better parallel load balance
    sizes = [json.loads(row.latest_dev_target_summary)["n"] for row in index.itertuples()]
    ordered = sorted(range(len(jobs)), key=lambda i: sizes[i], reverse=True)
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, jobs[i]): i for i in ordered}
        for fut, i in futures.items():
            results[i] = fut.result()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with the 11 official PPMI curated extracts.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS, help="Directory for aggregate output tables.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Private cache for participant-level arrays.")
    parser.add_argument("--pilot", action="store_true", help="Run 10-round audit and scikit-learn equivalence verification.")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()

    data_dir = reg.resolve_data_dir(args.data_dir)
    results_dir = args.results_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    identity = run_identity(data_dir)
    started = time.perf_counter()

    index, inv_summary = prepare_unified_dataset(data_dir, results_dir, cache_dir, identity)

    if args.pilot:
        fit_start = time.perf_counter()
        checks = run_all_tasks(index, cache_dir, PILOT_ROUNDS, identity, args.workers, pilot=True)
        fit_elapsed = time.perf_counter() - fit_start
        est_full = fit_elapsed * ROUNDS / PILOT_ROUNDS
        total_releases = sum(x["release_points"] for x in checks)
        total_stages = sum(x["checked_stages"] for x in checks)
        c23 = sum(1 for x in checks if x["first_cut"] == "20230612")
        save_json(
            cache_dir / "pilot.json",
            {
                "status": "passed",
                "identity": identity,
                "pilot_rounds": PILOT_ROUNDS,
                "workers": args.workers,
                "fit_elapsed_seconds": fit_elapsed,
                "estimated_full_seconds": est_full,
                "task_count": len(checks),
                "release_points": total_releases,
                "chronological_stages": total_stages,
                "tasks": checks,
            },
        )
        print(
            f"Pilot passed: {len(checks)} tasks, {total_releases} official release points, "
            f"{total_stages} verified stages ({c23} starting at 20230612); "
            f"fit={fit_elapsed:.1f}s, est_1000={est_full:.0f}s",
            flush=True,
        )
        return

    receipt = read_json(cache_dir / "pilot.json")
    assert receipt["identity"] == identity and receipt["status"] == "passed", "Run --pilot first after any code/data change."

    outputs = run_all_tasks(index, cache_dir, ROUNDS, identity, args.workers, pilot=False)
    releases_df = pd.DataFrame([row for out in outputs for row in out["releases"]])
    stages_df = pd.DataFrame([row for out in outputs for row in out["stages"]])
    steps_df = pd.DataFrame([row for out in outputs for row in out["steps"]])

    assert not releases_df.duplicated(["task", "version"]).any()
    assert not stages_df.duplicated(["task", "training_n"]).any()

    results_dir.mkdir(parents=True, exist_ok=True)
    releases_df.to_csv(results_dir / "real_version_trajectory.csv", encoding="utf-8-sig", index=False)
    stages_df.to_csv(results_dir / "chronological_size_trajectory.csv", encoding="utf-8-sig", index=False)
    steps_df.to_csv(results_dir / "chronological_step_summary.csv", encoding="utf-8-sig", index=False)

    save_json(
        results_dir / "run_summary.json",
        {
            "status": "complete",
            "identity": identity,
            "unified_partition": inv_summary["unified_partition"],
            "task_count": len(index),
            "official_release_points": len(releases_df),
            "chronological_stages": len(stages_df),
            "full_fit_stages": int(stages_df.status.eq("fit").sum()),
            "partial_missing_class_stages": int(stages_df.status.eq("partial_missing_class").sum()),
            "missing_training_class_stages": int(stages_df.status.eq("missing_training_class").sum()),
            "stages_at_20230612": int(stages_df.verified_cut.eq("20230612").sum()),
            "split_stages": int(stages_df.boundary_group_split.sum()),
            "followup_rule": "Unified scheduled visit filter (V[0-9]+) within 21-27 months across all longitudinal tasks",
            "partition_rule": "Single stratified 30% SHA-256 reservation by (reference_cohort, earliest_contiguous_cut)",
            "time_rule": "Earliest contiguous matching official extract through August 2026 (age tolerance 0.01 yr; exact elsewhere)",
            "tie_handling": f"{ROUNDS} nested within-snapshot permutations; earlier snapshots fixed",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "pilot": receipt,
        },
    )
    print(
        f"Complete: {len(index)} tasks; {len(releases_df)} official release points; "
        f"{len(stages_df)} chronological stages; {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
