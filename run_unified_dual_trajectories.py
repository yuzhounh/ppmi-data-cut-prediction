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
from scipy.optimize import brentq, minimize
from scipy.stats import rankdata
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss as sklearn_log_loss, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import ppmi_task_registry as reg

AGE_TOLERANCE = 0.01  # Years (~3.65 days): absorbs CSV 4-decimal rounding and 365/365.25 age divisor shifts.
SEED = 20261121
ROUNDS = 1000
PILOT_ROUNDS = 10
REFERENCE_TOLERANCE = 1e-7
SECONDARY_TOLERANCE = 1e-11
CALIBRATION_TOLERANCE = 1e-6
PROB_CLIP = 1e-15

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


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact tie-aware binary ROC AUC via average ranks (equivalent to Mann--Whitney U)."""
    pos = y_true == 1
    n_pos = int(np.count_nonzero(pos))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score, method="average")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def secondary_metrics(
    y: np.ndarray, pred: np.ndarray, num_classes: int, ss_tot: float | None = None
) -> tuple[float, float, float]:
    """Compute scale-free secondary benchmarks: AUROC (binary / macro-OvR) or R^2 + RMSE."""
    if num_classes == 2:
        return binary_auroc(y, pred[:, 1]), float("nan"), float("nan")
    if num_classes > 2:
        aucs = [binary_auroc((y == c).astype(int), pred[:, c]) for c in range(num_classes)]
        return float(np.mean(aucs)), float("nan"), float("nan")
    sq = (y - pred) ** 2
    rmse = float(math.sqrt(float(sq.mean())))
    denom = float(np.sum((y - y.mean()) ** 2)) if ss_tot is None else ss_tot
    r2 = float(1.0 - float(sq.sum()) / denom) if denom > 0 else float("nan")
    return float("nan"), r2, rmse


def reference_secondary_metrics(y: np.ndarray, pred: np.ndarray, num_classes: int) -> tuple[float, float, float]:
    """Reference scikit-learn secondary metrics used by the pilot audit and unit tests."""
    if num_classes == 2:
        return float(roc_auc_score(y, pred[:, 1])), float("nan"), float("nan")
    if num_classes > 2:
        return float(roc_auc_score(y, pred, multi_class="ovr", average="macro")), float("nan"), float("nan")
    return float("nan"), float(r2_score(y, pred)), float(root_mean_squared_error(y, pred))


def clip_probs(pred: np.ndarray) -> np.ndarray:
    """Clip predicted probabilities to [PROB_CLIP, 1-PROB_CLIP] and row-normalize."""
    p = np.clip(pred, PROB_CLIP, 1.0 - PROB_CLIP)
    return p / p.sum(axis=1, keepdims=True)


def binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact tie-aware step-function Average Precision matching scikit-learn."""
    n_pos = int(np.count_nonzero(y_true == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = (y_true[order] == 1).astype(np.float64)
    scores_sorted = y_score[order]
    distinct = np.flatnonzero(np.diff(scores_sorted))
    threshold_idx = np.concatenate((distinct, [len(scores_sorted) - 1]))
    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = (1.0 + threshold_idx) - tps
    precision = tps / (tps + fps)
    recall = tps / float(n_pos)
    prev_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev_recall) * precision))


def case_log_loss(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-participant negative log-likelihood after [PROB_CLIP, 1-PROB_CLIP] clipping."""
    p = clip_probs(pred)
    return -np.log(p[np.arange(len(y)), y])


def binary_citl(y: np.ndarray, p1: np.ndarray) -> float:
    """Fixed-slope-1 calibration-in-the-large solving mean(sigmoid(logit(p) + alpha)) = mean(y)."""
    y_mean = float(np.mean(y))
    if y_mean <= 0.0 or y_mean >= 1.0:
        return float("nan")
    pc = np.clip(p1, PROB_CLIP, 1.0 - PROB_CLIP)
    eta = np.log(pc / (1.0 - pc))
    p_mean = float(np.clip(pc.mean(), PROB_CLIP, 1.0 - PROB_CLIP))
    alpha = math.log(y_mean / (1.0 - y_mean)) - math.log(p_mean / (1.0 - p_mean))
    for _ in range(30):
        z = np.clip(eta + alpha, -40.0, 40.0)
        q = 1.0 / (1.0 + np.exp(-z))
        diff = float(q.mean()) - y_mean
        if abs(diff) < 1e-12:
            return float(alpha)
        grad = float(np.mean(q * (1.0 - q)))
        if grad < 1e-14:
            break
        step = diff / grad
        alpha -= max(-5.0, min(5.0, step))

    def f(a: float) -> float:
        z = np.clip(eta + a, -50.0, 50.0)
        return float(np.mean(1.0 / (1.0 + np.exp(-z)))) - y_mean

    try:
        return float(brentq(f, -40.0, 40.0, xtol=1e-12, rtol=1e-12))
    except ValueError:
        return float("nan")


def binary_cal_slope(y: np.ndarray, p1: np.ndarray) -> float:
    """Unpenalized logistic calibration slope b in logit(P(Y=1)) = a + b * logit(p)."""
    if y.min() == y.max():
        return float("nan")
    pc = np.clip(p1, PROB_CLIP, 1.0 - PROB_CLIP)
    eta = np.log(pc / (1.0 - pc))
    if float(np.std(eta)) < 1e-12:
        return float("nan")
    yf = y.astype(np.float64)
    a, b = 0.0, 1.0
    for _ in range(60):
        z = np.clip(a + b * eta, -40.0, 40.0)
        q = 1.0 / (1.0 + np.exp(-z))
        r = q - yf
        g0 = float(np.sum(r))
        g1 = float(np.sum(r * eta))
        if max(abs(g0), abs(g1)) < 1e-9:
            return float(b) if abs(b) < 50.0 else float("nan")
        w = q * (1.0 - q)
        h00 = float(np.sum(w))
        h01 = float(np.sum(w * eta))
        h11 = float(np.sum(w * eta * eta))
        det = h00 * h11 - h01 * h01
        if det <= 1e-14:
            return float("nan")
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        nll0 = float(np.sum(np.logaddexp(0.0, z) - yf * z))
        scale = 1.0
        while scale >= 1e-4:
            na, nb = a - scale * da, b - scale * db
            nz = np.clip(na + nb * eta, -40.0, 40.0)
            nll1 = float(np.sum(np.logaddexp(0.0, nz) - yf * nz))
            if nll1 <= nll0 + 1e-12:
                a, b = na, nb
                break
            scale *= 0.5
        else:
            return float("nan")
    return float("nan")


def reference_binary_cal_slope(y: np.ndarray, p1: np.ndarray) -> float:
    """Reference SciPy BFGS unpenalized calibration slope used by the pilot audit and unit tests."""
    if y.min() == y.max():
        return float("nan")
    pc = np.clip(p1, PROB_CLIP, 1.0 - PROB_CLIP)
    eta = np.log(pc / (1.0 - pc))
    if float(np.std(eta)) < 1e-12:
        return float("nan")
    yf = y.astype(np.float64)

    def obj_and_grad(theta: np.ndarray):
        z = np.clip(theta[0] + theta[1] * eta, -40.0, 40.0)
        q = 1.0 / (1.0 + np.exp(-z))
        nll = float(np.sum(np.logaddexp(0.0, z) - yf * z))
        r = q - yf
        return nll, np.array([float(np.sum(r)), float(np.sum(r * eta))], dtype=np.float64)

    res = minimize(obj_and_grad, np.array([0.0, 1.0]), jac=True, method="BFGS", options={"gtol": 1e-10})
    if res.success and abs(float(res.x[1])) < 50.0:
        return float(res.x[1])
    return float("nan")


def binary_quantile_ece(y_bin: np.ndarray, p1: np.ndarray, n_bins: int = 10) -> float:
    """10-bin equal-count (quantile) Expected Calibration Error."""
    n = len(y_bin)
    if n == 0:
        return float("nan")
    order = np.argsort(p1, kind="mergesort")
    bins = np.array_split(order, min(n_bins, n))
    ece = 0.0
    for b in bins:
        if len(b):
            ece += (len(b) / n) * abs(float(p1[b].mean()) - float(y_bin[b].mean()))
    return float(ece)


def prior_recalibrate(
    pred: np.ndarray,
    train_prior: np.ndarray | float,
    ref_prior: np.ndarray | float,
    num_classes: int,
) -> np.ndarray:
    """Zero-leakage closed-form prior-ratio (classification) or aggregate mean-shift (regression) alignment."""
    if num_classes >= 2:
        p = clip_probs(pred)
        tr = np.asarray(train_prior, dtype=np.float64)
        rf = np.asarray(ref_prior, dtype=np.float64)
        if np.any(tr <= 0.0) or np.any(rf <= 0.0):
            return np.full_like(p, np.nan)
        adj = p * (rf / tr)[None, :]
        return adj / adj.sum(axis=1, keepdims=True)
    return pred + (float(ref_prior) - float(train_prior))


@dataclass(frozen=True)
class DiagMetrics:
    auprc: float
    auprc_lift: float
    log_loss: float
    citl: float
    macro_abs_citl: float
    eo_ratio: float
    ece: float
    max_abs_prior_bias: float
    cal_me: float
    cal_slope: float
    medae: float
    nmae: float
    prior_adj_error: float
    prior_adj_log_loss: float
    prior_adj_r2: float
    prior_adj_rmse: float


def diagnostic_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    train_prior: np.ndarray | float,
    ref_prior: np.ndarray | float,
    ss_tot: float | None = None,
    eval_sd: float | None = None,
    compute_cls_slope: bool = False,
) -> DiagMetrics:
    """Compute M1 calibration, M2 AUPRC/log-loss, and M3 prior/mean-shift counterfactual metrics."""
    nan = float("nan")
    adj_pred = prior_recalibrate(pred, train_prior, ref_prior, num_classes)
    if num_classes == 2:
        p1 = pred[:, 1]
        ap = binary_auprc(y, p1)
        y_mean = float(np.mean(y))
        ap_lift = ap - y_mean
        ll = float(case_log_loss(y, pred).mean())
        citl = binary_citl(y, p1)
        eo = float(p1.mean() / y_mean) if y_mean > 0 else nan
        ece = binary_quantile_ece(y.astype(np.float64), p1)
        max_bias = abs(float(p1.mean()) - y_mean)
        slope = binary_cal_slope(y, p1) if compute_cls_slope else nan
        adj_err = float(case_loss(y, adj_pred, 2).mean()) if np.isfinite(adj_pred).all() else nan
        adj_ll = float(case_log_loss(y, adj_pred).mean()) if np.isfinite(adj_pred).all() else nan
        return DiagMetrics(
            auprc=ap,
            auprc_lift=ap_lift,
            log_loss=ll,
            citl=citl,
            macro_abs_citl=abs(citl) if math.isfinite(citl) else nan,
            eo_ratio=eo,
            ece=ece,
            max_abs_prior_bias=max_bias,
            cal_me=nan,
            cal_slope=slope,
            medae=nan,
            nmae=nan,
            prior_adj_error=adj_err,
            prior_adj_log_loss=adj_ll,
            prior_adj_r2=nan,
            prior_adj_rmse=nan,
        )
    if num_classes > 2:
        aps = [binary_auprc((y == c).astype(int), pred[:, c]) for c in range(num_classes)]
        ap = float(np.mean(aps))
        ap_lift = ap - (1.0 / num_classes)
        ll = float(case_log_loss(y, pred).mean())
        citls = [binary_citl((y == c).astype(int), pred[:, c]) for c in range(num_classes)]
        macro_abs_citl = float(np.mean([abs(v) for v in citls]))
        eces = [binary_quantile_ece((y == c).astype(np.float64), pred[:, c]) for c in range(num_classes)]
        ece = float(np.mean(eces))
        obs_shares = np.bincount(y, minlength=num_classes) / float(len(y))
        max_bias = float(np.max(np.abs(pred.mean(axis=0) - obs_shares)))
        adj_err = float(case_loss(y, adj_pred, num_classes).mean()) if np.isfinite(adj_pred).all() else nan
        adj_ll = float(case_log_loss(y, adj_pred).mean()) if np.isfinite(adj_pred).all() else nan
        return DiagMetrics(
            auprc=ap,
            auprc_lift=ap_lift,
            log_loss=ll,
            citl=nan,
            macro_abs_citl=macro_abs_citl,
            eo_ratio=nan,
            ece=ece,
            max_abs_prior_bias=max_bias,
            cal_me=nan,
            cal_slope=nan,
            medae=nan,
            nmae=nan,
            prior_adj_error=adj_err,
            prior_adj_log_loss=adj_ll,
            prior_adj_r2=nan,
            prior_adj_rmse=nan,
        )
    # Continuous regression
    abs_err = np.abs(y - pred)
    me = float(np.mean(pred - y))
    vp = float(np.var(pred))
    slope = float(np.mean((pred - pred.mean()) * (y - y.mean())) / vp) if vp > 1e-18 else nan
    medae = float(np.median(abs_err))
    denom = float(np.sum((y - y.mean()) ** 2)) if ss_tot is None else ss_tot
    sd = math.sqrt(denom / len(y)) if eval_sd is None else eval_sd
    nmae = float(abs_err.mean() / sd) if sd > 0 else nan
    adj_err = float(np.mean(np.abs(y - adj_pred)))
    adj_sq = (y - adj_pred) ** 2
    adj_rmse = float(math.sqrt(float(adj_sq.mean())))
    adj_r2 = float(1.0 - float(adj_sq.sum()) / denom) if denom > 0 else nan
    return DiagMetrics(
        auprc=nan,
        auprc_lift=nan,
        log_loss=nan,
        citl=nan,
        macro_abs_citl=nan,
        eo_ratio=nan,
        ece=nan,
        max_abs_prior_bias=nan,
        cal_me=me,
        cal_slope=slope,
        medae=medae,
        nmae=nmae,
        prior_adj_error=adj_err,
        prior_adj_log_loss=nan,
        prior_adj_r2=adj_r2,
        prior_adj_rmse=adj_rmse,
    )


def reference_diagnostic_metrics(
    y: np.ndarray, pred: np.ndarray, num_classes: int, compute_cls_slope: bool = False
) -> dict[str, float]:
    """Reference scikit-learn / SciPy diagnostic metrics used by the pilot audit and unit tests."""
    if num_classes == 2:
        ap = float(average_precision_score(y, pred[:, 1]))
        ll = float(sklearn_log_loss(y, clip_probs(pred), labels=[0, 1]))
        out = {"auprc": ap, "log_loss": ll}
        if compute_cls_slope:
            out["cal_slope"] = reference_binary_cal_slope(y, pred[:, 1])
        return out
    if num_classes > 2:
        aps = [float(average_precision_score((y == c).astype(int), pred[:, c])) for c in range(num_classes)]
        ll = float(sklearn_log_loss(y, clip_probs(pred), labels=list(range(num_classes))))
        return {"auprc": float(np.mean(aps)), "log_loss": ll}
    return {}


def eval_bootstrap_indices(eval_y: np.ndarray, task_id: str, k: int, n_boot: int) -> np.ndarray:
    """Stratified (classification) or IID (regression) evaluation-participant bootstrap indices (purpose=3)."""
    rng = random_stream(task_id, 0, purpose=3)
    n = len(eval_y)
    if k >= 2:
        class_indices = [np.flatnonzero(eval_y == c) for c in range(k)]
        parts = [rng.choice(idx, size=(n_boot, len(idx)), replace=True) for idx in class_indices]
        return np.concatenate(parts, axis=1)
    return rng.integers(0, n, size=(n_boot, n))


def _ci_bounds(vals: np.ndarray) -> tuple[float, float]:
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def paired_eval_bootstrap_release(
    eval_y: np.ndarray,
    pred_first: np.ndarray,
    pred_last: np.ndarray,
    adj_first: np.ndarray,
    adj_last: np.ndarray,
    k: int,
    idx: np.ndarray,
) -> dict[str, float]:
    """Paired evaluation-participant bootstrap 95% CIs for first-vs-latest release metrics."""
    nan = float("nan")
    loss_f = case_loss(eval_y, pred_first, k)
    loss_l = case_loss(eval_y, pred_last, k)
    adj_loss_f = case_loss(eval_y, adj_first, k)
    adj_loss_l = case_loss(eval_y, adj_last, k)

    err_diff_b = (loss_l - loss_f)[idx].mean(axis=1)
    adj_diff_b = (adj_loss_l - adj_loss_f)[idx].mean(axis=1)
    prior_comp_b = err_diff_b - adj_diff_b
    last_gain_b = (loss_l - adj_loss_l)[idx].mean(axis=1)

    out = {
        "rel_err_diff": float(loss_l.mean() - loss_f.mean()),
        "rel_err_ci_lo": _ci_bounds(err_diff_b)[0],
        "rel_err_ci_hi": _ci_bounds(err_diff_b)[1],
        "rel_adj_err_diff": float(adj_loss_l.mean() - adj_loss_f.mean()),
        "rel_adj_err_ci_lo": _ci_bounds(adj_diff_b)[0],
        "rel_adj_err_ci_hi": _ci_bounds(adj_diff_b)[1],
        "rel_prior_comp_diff": float((loss_l.mean() - loss_f.mean()) - (adj_loss_l.mean() - adj_loss_f.mean())),
        "rel_prior_comp_ci_lo": _ci_bounds(prior_comp_b)[0],
        "rel_prior_comp_ci_hi": _ci_bounds(prior_comp_b)[1],
        "rel_last_raw_minus_adj": float(loss_l.mean() - adj_loss_l.mean()),
        "rel_last_raw_minus_adj_ci_lo": _ci_bounds(last_gain_b)[0],
        "rel_last_raw_minus_adj_ci_hi": _ci_bounds(last_gain_b)[1],
    }

    n_boot = idx.shape[0]
    if k >= 2:
        ll_f = case_log_loss(eval_y, pred_first)
        ll_l = case_log_loss(eval_y, pred_last)
        adj_ll_f = case_log_loss(eval_y, adj_first)
        adj_ll_l = case_log_loss(eval_y, adj_last)
        ll_diff_b = (ll_l - ll_f)[idx].mean(axis=1)
        adj_ll_diff_b = (adj_ll_l - adj_ll_f)[idx].mean(axis=1)

        auc_b = np.empty(n_boot, dtype=np.float64)
        ap_b = np.empty(n_boot, dtype=np.float64)
        ece_b = np.empty(n_boot, dtype=np.float64)
        citl_b = np.full(n_boot, np.nan, dtype=np.float64)

        for b in range(n_boot):
            bi = idx[b]
            yb = eval_y[bi]
            pf = pred_first[bi]
            pl = pred_last[bi]
            if k == 2:
                auc_b[b] = binary_auroc(yb, pl[:, 1]) - binary_auroc(yb, pf[:, 1])
                ap_b[b] = binary_auprc(yb, pl[:, 1]) - binary_auprc(yb, pf[:, 1])
                ece_b[b] = binary_quantile_ece(yb.astype(np.float64), pl[:, 1]) - binary_quantile_ece(
                    yb.astype(np.float64), pf[:, 1]
                )
                citl_b[b] = binary_citl(yb, pl[:, 1]) - binary_citl(yb, pf[:, 1])
            else:
                auc_l = float(np.mean([binary_auroc((yb == c).astype(int), pl[:, c]) for c in range(k)]))
                auc_f = float(np.mean([binary_auroc((yb == c).astype(int), pf[:, c]) for c in range(k)]))
                ap_l = float(np.mean([binary_auprc((yb == c).astype(int), pl[:, c]) for c in range(k)]))
                ap_f = float(np.mean([binary_auprc((yb == c).astype(int), pf[:, c]) for c in range(k)]))
                ece_l = float(np.mean([binary_quantile_ece((yb == c).astype(np.float64), pl[:, c]) for c in range(k)]))
                ece_f = float(np.mean([binary_quantile_ece((yb == c).astype(np.float64), pf[:, c]) for c in range(k)]))
                auc_b[b] = auc_l - auc_f
                ap_b[b] = ap_l - ap_f
                ece_b[b] = ece_l - ece_f

        auc_pt = secondary_metrics(eval_y, pred_last, k)[0] - secondary_metrics(eval_y, pred_first, k)[0]
        if k == 2:
            ap_pt = binary_auprc(eval_y, pred_last[:, 1]) - binary_auprc(eval_y, pred_first[:, 1])
            ece_pt = binary_quantile_ece(eval_y.astype(np.float64), pred_last[:, 1]) - binary_quantile_ece(
                eval_y.astype(np.float64), pred_first[:, 1]
            )
            citl_pt = binary_citl(eval_y, pred_last[:, 1]) - binary_citl(eval_y, pred_first[:, 1])
        else:
            ap_pt = float(
                np.mean([binary_auprc((eval_y == c).astype(int), pred_last[:, c]) for c in range(k)])
                - np.mean([binary_auprc((eval_y == c).astype(int), pred_first[:, c]) for c in range(k)])
            )
            ece_pt = float(
                np.mean([binary_quantile_ece((eval_y == c).astype(np.float64), pred_last[:, c]) for c in range(k)])
                - np.mean([binary_quantile_ece((eval_y == c).astype(np.float64), pred_first[:, c]) for c in range(k)])
            )
            citl_pt = nan

        out.update(
            {
                "rel_auroc_diff": float(auc_pt),
                "rel_auroc_ci_lo": _ci_bounds(auc_b)[0],
                "rel_auroc_ci_hi": _ci_bounds(auc_b)[1],
                "rel_auprc_diff": float(ap_pt),
                "rel_auprc_ci_lo": _ci_bounds(ap_b)[0],
                "rel_auprc_ci_hi": _ci_bounds(ap_b)[1],
                "rel_logloss_diff": float(ll_l.mean() - ll_f.mean()),
                "rel_logloss_ci_lo": _ci_bounds(ll_diff_b)[0],
                "rel_logloss_ci_hi": _ci_bounds(ll_diff_b)[1],
                "rel_adj_logloss_diff": float(adj_ll_l.mean() - adj_ll_f.mean()),
                "rel_adj_logloss_ci_lo": _ci_bounds(adj_ll_diff_b)[0],
                "rel_adj_logloss_ci_hi": _ci_bounds(adj_ll_diff_b)[1],
                "rel_citl_diff": float(citl_pt),
                "rel_citl_ci_lo": _ci_bounds(citl_b)[0],
                "rel_citl_ci_hi": _ci_bounds(citl_b)[1],
                "rel_ece_diff": float(ece_pt),
                "rel_ece_ci_lo": _ci_bounds(ece_b)[0],
                "rel_ece_ci_hi": _ci_bounds(ece_b)[1],
                "rel_r2_diff": nan,
                "rel_r2_ci_lo": nan,
                "rel_r2_ci_hi": nan,
                "rel_rmse_diff": nan,
                "rel_rmse_ci_lo": nan,
                "rel_rmse_ci_hi": nan,
                "rel_me_diff": nan,
                "rel_me_ci_lo": nan,
                "rel_me_ci_hi": nan,
                "rel_medae_diff": nan,
                "rel_medae_ci_lo": nan,
                "rel_medae_ci_hi": nan,
                "rel_nmae_diff": nan,
                "rel_nmae_ci_lo": nan,
                "rel_nmae_ci_hi": nan,
            }
        )
    else:
        y_b = eval_y[idx]
        pf_b = pred_first[idx]
        pl_b = pred_last[idx]
        sq_f_b = (y_b - pf_b) ** 2
        sq_l_b = (y_b - pl_b) ** 2
        y_bar_b = y_b.mean(axis=1, keepdims=True)
        ss_tot_b = np.sum((y_b - y_bar_b) ** 2, axis=1)
        sd_b = np.sqrt(ss_tot_b / y_b.shape[1])
        r2_diff_b = (1.0 - sq_l_b.sum(axis=1) / ss_tot_b) - (1.0 - sq_f_b.sum(axis=1) / ss_tot_b)
        rmse_diff_b = np.sqrt(sq_l_b.mean(axis=1)) - np.sqrt(sq_f_b.mean(axis=1))
        me_diff_b = (pl_b - y_b).mean(axis=1) - (pf_b - y_b).mean(axis=1)
        medae_diff_b = np.median(np.abs(y_b - pl_b), axis=1) - np.median(np.abs(y_b - pf_b), axis=1)
        nmae_diff_b = (np.abs(y_b - pl_b).mean(axis=1) - np.abs(y_b - pf_b).mean(axis=1)) / sd_b

        ss_tot_pt = float(np.sum((eval_y - eval_y.mean()) ** 2))
        sd_pt = math.sqrt(ss_tot_pt / len(eval_y))
        _, r2_f, rmse_f = secondary_metrics(eval_y, pred_first, 0, ss_tot_pt)
        _, r2_l, rmse_l = secondary_metrics(eval_y, pred_last, 0, ss_tot_pt)
        out.update(
            {
                "rel_auroc_diff": nan,
                "rel_auroc_ci_lo": nan,
                "rel_auroc_ci_hi": nan,
                "rel_auprc_diff": nan,
                "rel_auprc_ci_lo": nan,
                "rel_auprc_ci_hi": nan,
                "rel_logloss_diff": nan,
                "rel_logloss_ci_lo": nan,
                "rel_logloss_ci_hi": nan,
                "rel_adj_logloss_diff": nan,
                "rel_adj_logloss_ci_lo": nan,
                "rel_adj_logloss_ci_hi": nan,
                "rel_citl_diff": nan,
                "rel_citl_ci_lo": nan,
                "rel_citl_ci_hi": nan,
                "rel_ece_diff": nan,
                "rel_ece_ci_lo": nan,
                "rel_ece_ci_hi": nan,
                "rel_r2_diff": float(r2_l - r2_f),
                "rel_r2_ci_lo": _ci_bounds(r2_diff_b)[0],
                "rel_r2_ci_hi": _ci_bounds(r2_diff_b)[1],
                "rel_rmse_diff": float(rmse_l - rmse_f),
                "rel_rmse_ci_lo": _ci_bounds(rmse_diff_b)[0],
                "rel_rmse_ci_hi": _ci_bounds(rmse_diff_b)[1],
                "rel_me_diff": float((pred_last - eval_y).mean() - (pred_first - eval_y).mean()),
                "rel_me_ci_lo": _ci_bounds(me_diff_b)[0],
                "rel_me_ci_hi": _ci_bounds(me_diff_b)[1],
                "rel_medae_diff": float(np.median(np.abs(eval_y - pred_last)) - np.median(np.abs(eval_y - pred_first))),
                "rel_medae_ci_lo": _ci_bounds(medae_diff_b)[0],
                "rel_medae_ci_hi": _ci_bounds(medae_diff_b)[1],
                "rel_nmae_diff": float((loss_l.mean() - loss_f.mean()) / sd_pt),
                "rel_nmae_ci_lo": _ci_bounds(nmae_diff_b)[0],
                "rel_nmae_ci_hi": _ci_bounds(nmae_diff_b)[1],
            }
        )
    return out


def label_counts(values: np.ndarray, nclasses: int) -> np.ndarray:
    return np.bincount(values, minlength=nclasses) if nclasses else np.empty(0, dtype=int)


def run_identity(data_dir: Path) -> dict:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(reg.__file__)):
        digest.update(path.read_bytes())
    input_digest = hashlib.sha256(Path(reg.__file__).read_bytes()).hexdigest()
    snapshots = {}
    for version in reg.VERSIONS:
        info = (data_dir / reg.extract_filename(version)).stat()
        snapshots[version] = [info.st_size, info.st_mtime_ns]
    return {
        "code_digest": digest.hexdigest(),
        "input_digest": input_digest,
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


def _inputs_match_cache(cached: dict, current: dict) -> bool:
    keys = ("snapshots", "reserve_seed", "reserve_fraction", "min_releases", "age_tolerance_years")
    if any(cached.get(k) != current.get(k) for k in keys):
        return False
    if "input_digest" in cached:
        return cached["input_digest"] == current["input_digest"]
    return True


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
        and _inputs_match_cache(read_json(marker)["identity"], identity)
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

    cohort_counts_by_release = {}
    for version in versions:
        c_counter = Counter(cohorts[version].values())
        cohort_counts_by_release[version] = {
            "total": int(len(cohorts[version])),
            "PD": int(c_counter.get(1, 0)),
            "HC": int(c_counter.get(2, 0)),
            "SWEDD": int(c_counter.get(3, 0)),
            "Prodromal": int(c_counter.get(4, 0)),
        }

    summary = {
        "versions": list(versions),
        "reference": reg.REFERENCE,
        "min_releases": reg.MIN_RELEASES,
        "cohort_counts_by_release": cohort_counts_by_release,
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


def _summarize_array(vals: np.ndarray) -> tuple[float, float, float, float, float]:
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    p025, med, p975 = np.percentile(finite, [2.5, 50.0, 97.5])
    return float(finite.mean()), float(finite.std(ddof=0)), float(p025), float(med), float(p975)


def _summarize_delta(diff: np.ndarray, direction: str) -> tuple[float, float, float, float, float]:
    finite = diff[np.isfinite(diff)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    d025, d50, d975 = np.percentile(finite, [2.5, 50.0, 97.5])
    prob = float((finite < 0).mean()) if direction == "down" else float((finite > 0).mean())
    return float(finite.mean()), float(d025), float(d50), float(d975), prob


def _prior_of(y: np.ndarray, k: int) -> np.ndarray | float:
    if k >= 2:
        return np.bincount(y, minlength=k).astype(np.float64) / float(len(y))
    return float(np.mean(y))


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
        ss_tot = float(np.sum((eval_y - eval_y.mean()) ** 2)) if k == 0 else None
        eval_sd = math.sqrt(ss_tot / len(eval_y)) if (k == 0 and ss_tot is not None) else None
        ref_prior = _prior_of(eval_y, k)
        secondary_label = "auroc" if k == 2 else ("macro_ovr_auroc" if k > 2 else "r2_rmse")
        max_ref_diff = 0.0
        max_sec_diff = 0.0
        max_cal_diff = 0.0

        # Trajectory 1: official curated releases
        release_rows = []
        pred_first = None
        pred_last = None
        adj_first = None
        adj_last = None
        started = time.perf_counter()
        for cut_idx, cut in enumerate(cuts):
            x_cut, y_cut = block[f"x_{cut}"], block[f"y_{cut}"]
            train_prior = _prior_of(y_cut, k)
            pred = predict(x_cut, y_cut, eval_x, k)
            loss = case_loss(eval_y, pred, k)
            assert np.isfinite(loss).all()
            auroc, r2, rmse = secondary_metrics(eval_y, pred, k, ss_tot)
            diag = diagnostic_metrics(
                eval_y,
                pred,
                k,
                train_prior,
                ref_prior,
                ss_tot=ss_tot,
                eval_sd=eval_sd,
                compute_cls_slope=True,
            )
            adj_pred = prior_recalibrate(pred, train_prior, ref_prior, k)
            if cut_idx == 0:
                pred_first, adj_first = pred, adj_pred
            if cut_idx == len(cuts) - 1:
                pred_last, adj_last = pred, adj_pred
            if pilot:
                direct = reference_predict(x_cut, y_cut, eval_x, k)
                max_ref_diff = max(max_ref_diff, float(np.max(np.abs(direct - pred))))
                ref_auc, ref_r2, ref_rmse = reference_secondary_metrics(eval_y, pred, k)
                for a, b in ((auroc, ref_auc), (r2, ref_r2), (rmse, ref_rmse)):
                    if math.isfinite(a) or math.isfinite(b):
                        max_sec_diff = max(max_sec_diff, abs(a - b))
                ref_diag = reference_diagnostic_metrics(eval_y, pred, k, compute_cls_slope=True)
                if "auprc" in ref_diag:
                    max_sec_diff = max(max_sec_diff, abs(diag.auprc - ref_diag["auprc"]))
                if "log_loss" in ref_diag:
                    max_sec_diff = max(max_sec_diff, abs(diag.log_loss - ref_diag["log_loss"]))
                if "cal_slope" in ref_diag and math.isfinite(ref_diag["cal_slope"]) and math.isfinite(diag.cal_slope):
                    max_cal_diff = max(max_cal_diff, abs(diag.cal_slope - ref_diag["cal_slope"]))
                if k == 2 and math.isfinite(diag.citl):
                    pc = np.clip(pred[:, 1], PROB_CLIP, 1.0 - PROB_CLIP)
                    eta = np.log(pc / (1.0 - pc))
                    citl_res = abs(float(np.mean(1.0 / (1.0 + np.exp(-np.clip(eta + diag.citl, -40.0, 40.0))))) - float(eval_y.mean()))
                    max_cal_diff = max(max_cal_diff, citl_res)
                    # Verify binary Bayes prior recalibration preserves AUROC and AUPRC exactly
                    max_sec_diff = max(max_sec_diff, abs(binary_auroc(eval_y, adj_pred[:, 1]) - auroc))
                    max_sec_diff = max(max_sec_diff, abs(binary_auprc(eval_y, adj_pred[:, 1]) - diag.auprc))
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
                    "secondary_metric": secondary_label,
                    "auroc": auroc,
                    "r2": r2,
                    "rmse": rmse,
                    "auprc": diag.auprc,
                    "auprc_lift": diag.auprc_lift,
                    "log_loss": diag.log_loss,
                    "citl": diag.citl,
                    "macro_abs_citl": diag.macro_abs_citl,
                    "eo_ratio": diag.eo_ratio,
                    "ece": diag.ece,
                    "max_abs_prior_bias": diag.max_abs_prior_bias,
                    "cal_me": diag.cal_me,
                    "cal_slope": diag.cal_slope,
                    "medae": diag.medae,
                    "nmae": diag.nmae,
                    "prior_adj_error": diag.prior_adj_error,
                    "prior_adj_log_loss": diag.prior_adj_log_loss,
                    "prior_adj_r2": diag.prior_adj_r2,
                    "prior_adj_rmse": diag.prior_adj_rmse,
                }
            )
        expected_anchor = release_rows[-1]["observed_release_error"]
        expected_anchor_sec = (release_rows[-1]["auroc"], release_rows[-1]["r2"], release_rows[-1]["rmse"])
        expected_anchor_diag = (
            release_rows[-1]["auprc"],
            release_rows[-1]["log_loss"],
            release_rows[-1]["citl"],
            release_rows[-1]["eo_ratio"],
            release_rows[-1]["ece"],
            release_rows[-1]["cal_me"],
            release_rows[-1]["cal_slope"],
            release_rows[-1]["medae"],
            release_rows[-1]["nmae"],
            release_rows[-1]["prior_adj_error"],
            release_rows[-1]["prior_adj_log_loss"],
            release_rows[-1]["prior_adj_r2"],
        )

        # Trajectory 2: chronological doubling within August 2026 reference
        stages = checked_stages(block, versions, timing)
        x, y = block["chrono_x"], block["chrono_y"]
        cut_rank = block["cutoff_idx"]
        n_stages = len(stages)
        assert stages[-1]["training_n"] == release_rows[-1]["training_n"]

        scores = np.full((rounds, n_stages), np.nan)
        aurocs = np.full((rounds, n_stages), np.nan)
        r2s = np.full((rounds, n_stages), np.nan)
        rmses = np.full((rounds, n_stages), np.nan)
        auprcs = np.full((rounds, n_stages), np.nan)
        loglosses = np.full((rounds, n_stages), np.nan)
        citls = np.full((rounds, n_stages), np.nan)
        eos = np.full((rounds, n_stages), np.nan)
        eces = np.full((rounds, n_stages), np.nan)
        cal_mes = np.full((rounds, n_stages), np.nan)
        cal_slopes = np.full((rounds, n_stages), np.nan)
        medaes = np.full((rounds, n_stages), np.nan)
        nmaes = np.full((rounds, n_stages), np.nan)
        adj_errors = np.full((rounds, n_stages), np.nan)
        adj_loglosses = np.full((rounds, n_stages), np.nan)
        adj_r2s = np.full((rounds, n_stages), np.nan)
        class_totals = np.zeros((rounds, n_stages, k), dtype=int) if k else np.zeros((rounds, n_stages, 0), dtype=int)
        split_loss_sum = np.zeros((n_stages, evaluation_n), dtype=np.float64)
        split_adj_loss_sum = np.zeros((n_stages, evaluation_n), dtype=np.float64)
        split_valid_cnt = np.zeros(n_stages, dtype=int)

        unsplit_cache = {}
        unsplit_losses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for col, stage in enumerate(stages):
            if not stage["boundary_group_split"]:
                n = stage["training_n"]
                sub_y = y[:n]
                cnt = label_counts(sub_y, k)
                if k and not cnt.all():
                    unsplit_cache[col] = (
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        DiagMetrics(*([np.nan] * 16)),
                        cnt,
                    )
                else:
                    tr_prior = _prior_of(sub_y, k)
                    pred = predict(x[:n], sub_y, eval_x, k)
                    loss = case_loss(eval_y, pred, k)
                    assert np.isfinite(loss).all()
                    auroc, r2, rmse = secondary_metrics(eval_y, pred, k, ss_tot)
                    diag = diagnostic_metrics(
                        eval_y,
                        pred,
                        k,
                        tr_prior,
                        ref_prior,
                        ss_tot=ss_tot,
                        eval_sd=eval_sd,
                        compute_cls_slope=True,
                    )
                    adj_pred = prior_recalibrate(pred, tr_prior, ref_prior, k)
                    adj_loss = case_loss(eval_y, adj_pred, k)
                    if pilot:
                        direct = reference_predict(x[:n], sub_y, eval_x, k)
                        max_ref_diff = max(max_ref_diff, float(np.max(np.abs(direct - pred))))
                        ref_auc, ref_r2, ref_rmse = reference_secondary_metrics(eval_y, pred, k)
                        for a, b in ((auroc, ref_auc), (r2, ref_r2), (rmse, ref_rmse)):
                            if math.isfinite(a) or math.isfinite(b):
                                max_sec_diff = max(max_sec_diff, abs(a - b))
                        ref_diag = reference_diagnostic_metrics(eval_y, pred, k, compute_cls_slope=True)
                        if "auprc" in ref_diag:
                            max_sec_diff = max(max_sec_diff, abs(diag.auprc - ref_diag["auprc"]))
                        if "log_loss" in ref_diag:
                            max_sec_diff = max(max_sec_diff, abs(diag.log_loss - ref_diag["log_loss"]))
                        if "cal_slope" in ref_diag and math.isfinite(ref_diag["cal_slope"]) and math.isfinite(diag.cal_slope):
                            max_cal_diff = max(max_cal_diff, abs(diag.cal_slope - ref_diag["cal_slope"]))
                    unsplit_cache[col] = (float(loss.mean()), auroc, r2, rmse, diag, cnt)
                    unsplit_losses[col] = (loss, adj_loss)

        assert n_stages - 1 in unsplit_cache
        anchor_tuple = unsplit_cache[n_stages - 1]
        assert np.isclose(anchor_tuple[0], expected_anchor, rtol=0, atol=1e-9), (
            key,
            anchor_tuple[0],
            expected_anchor,
        )
        for got_s, exp_s in zip(anchor_tuple[1:4], expected_anchor_sec):
            if math.isfinite(exp_s):
                assert np.isclose(got_s, exp_s, rtol=0, atol=1e-9), (key, got_s, exp_s)
        anchor_diag: DiagMetrics = anchor_tuple[4]
        got_diag_vec = (
            anchor_diag.auprc,
            anchor_diag.log_loss,
            anchor_diag.citl,
            anchor_diag.eo_ratio,
            anchor_diag.ece,
            anchor_diag.cal_me,
            anchor_diag.cal_slope,
            anchor_diag.medae,
            anchor_diag.nmae,
            anchor_diag.prior_adj_error,
            anchor_diag.prior_adj_log_loss,
            anchor_diag.prior_adj_r2,
        )
        for got_d, exp_d in zip(got_diag_vec, expected_anchor_diag):
            if math.isfinite(exp_d):
                assert np.isclose(got_d, exp_d, rtol=0, atol=1e-9), (key, got_d, exp_d)

        ckpt_path = cache_dir / f"{key}_tie_checkpoint.npz"
        start = 0
        if not pilot and ckpt_path.exists():
            with np.load(ckpt_path) as ckpt:
                if (
                    "auprcs" in ckpt
                    and "split_loss_sum" in ckpt
                    and ckpt["identity"].item() == identity["code_digest"]
                    and ckpt["scores"].shape[1] == n_stages
                ):
                    start = min(rounds, int(ckpt["completed"].item()))
                    scores[:start] = ckpt["scores"][:start]
                    aurocs[:start] = ckpt["aurocs"][:start]
                    r2s[:start] = ckpt["r2s"][:start]
                    rmses[:start] = ckpt["rmses"][:start]
                    auprcs[:start] = ckpt["auprcs"][:start]
                    loglosses[:start] = ckpt["loglosses"][:start]
                    citls[:start] = ckpt["citls"][:start]
                    eos[:start] = ckpt["eos"][:start]
                    eces[:start] = ckpt["eces"][:start]
                    cal_mes[:start] = ckpt["cal_mes"][:start]
                    cal_slopes[:start] = ckpt["cal_slopes"][:start]
                    medaes[:start] = ckpt["medaes"][:start]
                    nmaes[:start] = ckpt["nmaes"][:start]
                    adj_errors[:start] = ckpt["adj_errors"][:start]
                    adj_loglosses[:start] = ckpt["adj_loglosses"][:start]
                    adj_r2s[:start] = ckpt["adj_r2s"][:start]
                    split_loss_sum[:] = ckpt["split_loss_sum"]
                    split_adj_loss_sum[:] = ckpt["split_adj_loss_sum"]
                    split_valid_cnt[:] = ckpt["split_valid_cnt"]
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
                    val, auc_v, r2_v, rmse_v, d_v, cnt = unsplit_cache[col]
                    scores[r, col] = val
                    aurocs[r, col] = auc_v
                    r2s[r, col] = r2_v
                    rmses[r, col] = rmse_v
                    auprcs[r, col] = d_v.auprc
                    loglosses[r, col] = d_v.log_loss
                    citls[r, col] = d_v.citl
                    eos[r, col] = d_v.eo_ratio
                    eces[r, col] = d_v.ece
                    cal_mes[r, col] = d_v.cal_me
                    cal_slopes[r, col] = d_v.cal_slope
                    medaes[r, col] = d_v.medae
                    nmaes[r, col] = d_v.nmae
                    adj_errors[r, col] = d_v.prior_adj_error
                    adj_loglosses[r, col] = d_v.prior_adj_log_loss
                    adj_r2s[r, col] = d_v.prior_adj_r2
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
                tr_prior = (cnt.astype(np.float64) / float(n)) if k >= 2 else float(np.mean(sub_y))
                pred = predict(sub_x, sub_y, eval_x, k)
                loss = case_loss(eval_y, pred, k)
                assert np.isfinite(loss).all()
                auroc, r2, rmse = secondary_metrics(eval_y, pred, k, ss_tot)
                diag = diagnostic_metrics(
                    eval_y,
                    pred,
                    k,
                    tr_prior,
                    ref_prior,
                    ss_tot=ss_tot,
                    eval_sd=eval_sd,
                    compute_cls_slope=False,
                )
                adj_pred = prior_recalibrate(pred, tr_prior, ref_prior, k)
                adj_loss = case_loss(eval_y, adj_pred, k)
                split_loss_sum[col] += loss
                split_adj_loss_sum[col] += adj_loss
                split_valid_cnt[col] += 1
                if pilot and col in check_cols:
                    direct = reference_predict(sub_x, sub_y, eval_x, k)
                    max_ref_diff = max(max_ref_diff, float(np.max(np.abs(direct - pred))))
                    ref_auc, ref_r2, ref_rmse = reference_secondary_metrics(eval_y, pred, k)
                    for a, b in ((auroc, ref_auc), (r2, ref_r2), (rmse, ref_rmse)):
                        if math.isfinite(a) or math.isfinite(b):
                            max_sec_diff = max(max_sec_diff, abs(a - b))
                    ref_diag = reference_diagnostic_metrics(eval_y, pred, k, compute_cls_slope=False)
                    if "auprc" in ref_diag:
                        max_sec_diff = max(max_sec_diff, abs(diag.auprc - ref_diag["auprc"]))
                    if "log_loss" in ref_diag:
                        max_sec_diff = max(max_sec_diff, abs(diag.log_loss - ref_diag["log_loss"]))
                scores[r, col] = float(loss.mean())
                aurocs[r, col] = auroc
                r2s[r, col] = r2
                rmses[r, col] = rmse
                auprcs[r, col] = diag.auprc
                loglosses[r, col] = diag.log_loss
                citls[r, col] = diag.citl
                eos[r, col] = diag.eo_ratio
                eces[r, col] = diag.ece
                cal_mes[r, col] = diag.cal_me
                cal_slopes[r, col] = diag.cal_slope
                medaes[r, col] = diag.medae
                nmaes[r, col] = diag.nmae
                adj_errors[r, col] = diag.prior_adj_error
                adj_loglosses[r, col] = diag.prior_adj_log_loss
                adj_r2s[r, col] = diag.prior_adj_r2
            if not pilot and ((r + 1) % 250 == 0 or r + 1 == rounds):
                save_arrays(
                    ckpt_path,
                    identity=identity["code_digest"],
                    completed=r + 1,
                    scores=scores[: r + 1],
                    aurocs=aurocs[: r + 1],
                    r2s=r2s[: r + 1],
                    rmses=rmses[: r + 1],
                    auprcs=auprcs[: r + 1],
                    loglosses=loglosses[: r + 1],
                    citls=citls[: r + 1],
                    eos=eos[: r + 1],
                    eces=eces[: r + 1],
                    cal_mes=cal_mes[: r + 1],
                    cal_slopes=cal_slopes[: r + 1],
                    medaes=medaes[: r + 1],
                    nmaes=nmaes[: r + 1],
                    adj_errors=adj_errors[: r + 1],
                    adj_loglosses=adj_loglosses[: r + 1],
                    adj_r2s=adj_r2s[: r + 1],
                    split_loss_sum=split_loss_sum,
                    split_adj_loss_sum=split_adj_loss_sum,
                    split_valid_cnt=split_valid_cnt,
                    class_totals=class_totals[: r + 1],
                    max_ref_diff=max_ref_diff,
                )

        # M4: Paired evaluation-participant bootstrap (1,000 stratified/IID draws, purpose=3)
        assert pred_first is not None and pred_last is not None and adj_first is not None and adj_last is not None
        boot_idx = eval_bootstrap_indices(eval_y, key, k, n_boot=rounds)
        if pilot:
            # Verify exact zero-difference bootstrap invariance when both endpoints are identical
            zero_check = paired_eval_bootstrap_release(eval_y, pred_last, pred_last, pred_last, pred_last, k, boot_idx)
            for mk, mv in zero_check.items():
                if math.isfinite(mv):
                    assert abs(mv) < 1e-12, (key, mk, mv)

        boot_row = paired_eval_bootstrap_release(eval_y, pred_first, pred_last, adj_first, adj_last, k, boot_idx)

        # Assemble per-participant tie-averaged loss vectors for chronological M4 paired CIs
        stage_case_losses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for col in range(n_stages):
            if col in unsplit_losses:
                stage_case_losses[col] = unsplit_losses[col]
            elif split_valid_cnt[col] > 0:
                stage_case_losses[col] = (
                    split_loss_sum[col] / float(split_valid_cnt[col]),
                    split_adj_loss_sum[col] / float(split_valid_cnt[col]),
                )
        estimable_cols = sorted(stage_case_losses)
        full_col = n_stages - 1
        assert full_col in stage_case_losses
        full_loss_vec, full_adj_vec = stage_case_losses[full_col]

        if len(estimable_cols) >= 2:
            earliest_col = estimable_cols[0]
            penult_col = estimable_cols[-2]
            ear_loss_vec, ear_adj_vec = stage_case_losses[earliest_col]
            pen_loss_vec, pen_adj_vec = stage_case_losses[penult_col]
            step_b = (full_loss_vec - pen_loss_vec)[boot_idx].mean(axis=1)
            step_adj_b = (full_adj_vec - pen_adj_vec)[boot_idx].mean(axis=1)
            span_b = (full_loss_vec - ear_loss_vec)[boot_idx].mean(axis=1)
            span_adj_b = (full_adj_vec - ear_adj_vec)[boot_idx].mean(axis=1)
            chrono_boot = {
                "earliest_n": int(stages[earliest_col]["training_n"]),
                "penultimate_n": int(stages[penult_col]["training_n"]),
                "full_n": int(stages[full_col]["training_n"]),
                "step_err_diff": float((full_loss_vec - pen_loss_vec).mean()),
                "step_err_ci_lo": _ci_bounds(step_b)[0],
                "step_err_ci_hi": _ci_bounds(step_b)[1],
                "step_adj_err_diff": float((full_adj_vec - pen_adj_vec).mean()),
                "step_adj_err_ci_lo": _ci_bounds(step_adj_b)[0],
                "step_adj_err_ci_hi": _ci_bounds(step_adj_b)[1],
                "span_err_diff": float((full_loss_vec - ear_loss_vec).mean()),
                "span_err_ci_lo": _ci_bounds(span_b)[0],
                "span_err_ci_hi": _ci_bounds(span_b)[1],
                "span_adj_err_diff": float((full_adj_vec - ear_adj_vec).mean()),
                "span_adj_err_ci_lo": _ci_bounds(span_adj_b)[0],
                "span_adj_err_ci_hi": _ci_bounds(span_adj_b)[1],
            }
        else:
            nan = float("nan")
            chrono_boot = {
                "earliest_n": None,
                "penultimate_n": None,
                "full_n": int(stages[full_col]["training_n"]),
                "step_err_diff": nan,
                "step_err_ci_lo": nan,
                "step_err_ci_hi": nan,
                "step_adj_err_diff": nan,
                "step_adj_err_ci_lo": nan,
                "step_adj_err_ci_hi": nan,
                "span_err_diff": nan,
                "span_err_ci_lo": nan,
                "span_err_ci_hi": nan,
                "span_adj_err_diff": nan,
                "span_adj_err_ci_lo": nan,
                "span_adj_err_ci_hi": nan,
            }

        raw_rel_arr = np.asarray([r["observed_release_error"] for r in release_rows], dtype=np.float64)
        adj_rel_arr = np.asarray([r["prior_adj_error"] for r in release_rows], dtype=np.float64)
        raw_spread_pct = float(100.0 * np.ptp(raw_rel_arr) / raw_rel_arr[0])
        adj_spread_pct = float(100.0 * np.ptp(adj_rel_arr) / adj_rel_arr[0])
        raw_delta = float(raw_rel_arr[-1] - raw_rel_arr[0])
        adj_delta = float(adj_rel_arr[-1] - adj_rel_arr[0])
        prior_explained_pct = float(100.0 * (raw_delta - adj_delta) / raw_delta) if raw_delta > 1e-12 else float("nan")

        paired_row = {
            "task": key,
            "primary_metric": primary_metric,
            "evaluation_n": evaluation_n,
            "first_version": cuts[0],
            "latest_version": cuts[-1],
            "first_raw_err": float(raw_rel_arr[0]),
            "last_raw_err": float(raw_rel_arr[-1]),
            "first_adj_err": float(adj_rel_arr[0]),
            "last_adj_err": float(adj_rel_arr[-1]),
            "raw_release_spread_pct": raw_spread_pct,
            "adj_release_spread_pct": adj_spread_pct,
            "prior_explained_pct": prior_explained_pct,
            "boot_rounds": rounds,
            **boot_row,
            **chrono_boot,
        }

        elapsed = time.perf_counter() - started
        if pilot:
            assert max_ref_diff < REFERENCE_TOLERANCE, (key, max_ref_diff)
            assert max_sec_diff < SECONDARY_TOLERANCE, (key, max_sec_diff)
            assert max_cal_diff < CALIBRATION_TOLERANCE, (key, max_cal_diff)
            return {
                "task": key,
                "release_points": len(release_rows),
                "checked_stages": n_stages,
                "split_stages": len(split_cols),
                "first_cut": stages[0]["verified_cut"],
                "last_cut": stages[-1]["verified_cut"],
                "max_reference_prediction_error": max_ref_diff,
                "max_secondary_metric_error": max_sec_diff,
                "max_calibration_metric_error": max_cal_diff,
                "fit_seconds": elapsed,
                "missing_class_draws": int(np.isnan(scores).sum()),
            }

        stage_rows = []
        for col, stage in enumerate(stages):
            vals = scores[:, col]
            finite = vals[np.isfinite(vals)]
            valid = len(finite)
            missing = rounds - valid
            mean_err, sd_err, p025, med, p975 = _summarize_array(vals)
            if col in stage_case_losses:
                assert np.isclose(float(stage_case_losses[col][0].mean()), mean_err, rtol=0, atol=1e-10), (key, col)
                adj_mean_check = _summarize_array(adj_errors[:, col])[0]
                assert np.isclose(float(stage_case_losses[col][1].mean()), adj_mean_check, rtol=0, atol=1e-10), (key, col)
            auc_mean, auc_sd, auc_p025, auc_med, auc_p975 = _summarize_array(aurocs[:, col])
            r2_mean, r2_sd, r2_p025, r2_med, r2_p975 = _summarize_array(r2s[:, col])
            rmse_mean, rmse_sd, rmse_p025, rmse_med, rmse_p975 = _summarize_array(rmses[:, col])
            ap_m, _, ap_lo, _, ap_hi = _summarize_array(auprcs[:, col])
            ll_m, _, ll_lo, _, ll_hi = _summarize_array(loglosses[:, col])
            ci_m, _, ci_lo, _, ci_hi = _summarize_array(citls[:, col])
            eo_m, _, eo_lo, _, eo_hi = _summarize_array(eos[:, col])
            ec_m, _, ec_lo, _, ec_hi = _summarize_array(eces[:, col])
            me_m, _, me_lo, _, me_hi = _summarize_array(cal_mes[:, col])
            sl_m, _, sl_lo, _, sl_hi = _summarize_array(cal_slopes[:, col])
            md_m, _, md_lo, _, md_hi = _summarize_array(medaes[:, col])
            nm_m, _, nm_lo, _, nm_hi = _summarize_array(nmaes[:, col])
            ae_m, _, ae_lo, _, ae_hi = _summarize_array(adj_errors[:, col])
            al_m, _, al_lo, _, al_hi = _summarize_array(adj_loglosses[:, col])
            ar_m, _, ar_lo, _, ar_hi = _summarize_array(adj_r2s[:, col])
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
                    "error_p025": p025,
                    "error_median": med,
                    "error_p975": p975,
                    "attempted_rounds": rounds,
                    "valid_rounds": valid,
                    "missing_class_rounds": missing,
                    "class_counts": json.dumps(mean_counts),
                    "status": status,
                    "auroc_mean": auc_mean,
                    "auroc_sd": auc_sd,
                    "auroc_p025": auc_p025,
                    "auroc_median": auc_med,
                    "auroc_p975": auc_p975,
                    "r2_mean": r2_mean,
                    "r2_sd": r2_sd,
                    "r2_p025": r2_p025,
                    "r2_median": r2_med,
                    "r2_p975": r2_p975,
                    "rmse_mean": rmse_mean,
                    "rmse_sd": rmse_sd,
                    "rmse_p025": rmse_p025,
                    "rmse_median": rmse_med,
                    "rmse_p975": rmse_p975,
                    "auprc_mean": ap_m,
                    "auprc_p025": ap_lo,
                    "auprc_p975": ap_hi,
                    "log_loss_mean": ll_m,
                    "log_loss_p025": ll_lo,
                    "log_loss_p975": ll_hi,
                    "citl_mean": ci_m,
                    "citl_p025": ci_lo,
                    "citl_p975": ci_hi,
                    "eo_ratio_mean": eo_m,
                    "eo_ratio_p025": eo_lo,
                    "eo_ratio_p975": eo_hi,
                    "ece_mean": ec_m,
                    "ece_p025": ec_lo,
                    "ece_p975": ec_hi,
                    "cal_me_mean": me_m,
                    "cal_me_p025": me_lo,
                    "cal_me_p975": me_hi,
                    "cal_slope_mean": sl_m,
                    "cal_slope_p025": sl_lo,
                    "cal_slope_p975": sl_hi,
                    "medae_mean": md_m,
                    "medae_p025": md_lo,
                    "medae_p975": md_hi,
                    "nmae_mean": nm_m,
                    "nmae_p025": nm_lo,
                    "nmae_p975": nm_hi,
                    "prior_adj_error_mean": ae_m,
                    "prior_adj_error_p025": ae_lo,
                    "prior_adj_error_p975": ae_hi,
                    "prior_adj_log_loss_mean": al_m,
                    "prior_adj_log_loss_p025": al_lo,
                    "prior_adj_log_loss_p975": al_hi,
                    "prior_adj_r2_mean": ar_m,
                    "prior_adj_r2_p025": ar_lo,
                    "prior_adj_r2_p975": ar_hi,
                }
            )

        step_rows = []
        for col in range(1, n_stages):
            diff = scores[:, col] - scores[:, col - 1]
            valid = int(np.isfinite(diff).sum())
            dmean, d025, d50, d975, p_dec = _summarize_delta(diff, "down")
            auc_dm, auc_d025, auc_d50, auc_d975, auc_pinc = _summarize_delta(aurocs[:, col] - aurocs[:, col - 1], "up")
            r2_dm, r2_d025, r2_d50, r2_d975, r2_pinc = _summarize_delta(r2s[:, col] - r2s[:, col - 1], "up")
            rmse_dm, rmse_d025, rmse_d50, rmse_d975, rmse_pdec = _summarize_delta(rmses[:, col] - rmses[:, col - 1], "down")
            adj_dm, adj_d025, adj_d50, adj_d975, adj_pdec = _summarize_delta(
                adj_errors[:, col] - adj_errors[:, col - 1], "down"
            )
            ap_dm, ap_d025, _, ap_d975, _ = _summarize_delta(auprcs[:, col] - auprcs[:, col - 1], "up")
            ll_dm, ll_d025, _, ll_d975, _ = _summarize_delta(loglosses[:, col] - loglosses[:, col - 1], "down")
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
                    "delta_p025": d025,
                    "delta_median": d50,
                    "delta_p975": d975,
                    "prob_decreased": p_dec,
                    "auroc_delta_mean": auc_dm,
                    "auroc_delta_p025": auc_d025,
                    "auroc_delta_median": auc_d50,
                    "auroc_delta_p975": auc_d975,
                    "auroc_prob_increased": auc_pinc,
                    "r2_delta_mean": r2_dm,
                    "r2_delta_p025": r2_d025,
                    "r2_delta_median": r2_d50,
                    "r2_delta_p975": r2_d975,
                    "r2_prob_increased": r2_pinc,
                    "rmse_delta_mean": rmse_dm,
                    "rmse_delta_p025": rmse_d025,
                    "rmse_delta_median": rmse_d50,
                    "rmse_delta_p975": rmse_d975,
                    "rmse_prob_decreased": rmse_pdec,
                    "prior_adj_delta_mean": adj_dm,
                    "prior_adj_delta_p025": adj_d025,
                    "prior_adj_delta_median": adj_d50,
                    "prior_adj_delta_p975": adj_d975,
                    "prior_adj_prob_decreased": adj_pdec,
                    "auprc_delta_mean": ap_dm,
                    "auprc_delta_p025": ap_d025,
                    "auprc_delta_p975": ap_d975,
                    "log_loss_delta_mean": ll_dm,
                    "log_loss_delta_p025": ll_d025,
                    "log_loss_delta_p975": ll_d975,
                }
            )
        print(f"{key} {rounds}r {elapsed:.2f}s (cuts={len(cuts)}, splits={len(split_cols)})", flush=True)
        return {
            "task": key,
            "releases": release_rows,
            "stages": stage_rows,
            "steps": step_rows,
            "paired": paired_row,
            "fit_seconds": elapsed,
            "max_reference_prediction_error": max_ref_diff,
            "max_secondary_metric_error": max_sec_diff,
            "max_calibration_metric_error": max_cal_diff,
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
        max_ref = max(x["max_reference_prediction_error"] for x in checks)
        max_sec = max(x["max_secondary_metric_error"] for x in checks)
        max_cal = max(x["max_calibration_metric_error"] for x in checks)
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
                "max_reference_prediction_error": max_ref,
                "max_secondary_metric_error": max_sec,
                "max_calibration_metric_error": max_cal,
                "tasks": checks,
            },
        )
        print(
            f"Pilot passed: {len(checks)} tasks, {total_releases} official release points, "
            f"{total_stages} verified stages ({c23} starting at 20230612); "
            f"max_ref={max_ref:.2e}, max_sec={max_sec:.2e}, max_cal={max_cal:.2e}; "
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
    paired_df = pd.DataFrame([out["paired"] for out in outputs])

    assert not releases_df.duplicated(["task", "version"]).any()
    assert not stages_df.duplicated(["task", "training_n"]).any()
    assert not paired_df.duplicated(["task"]).any()

    results_dir.mkdir(parents=True, exist_ok=True)
    releases_df.to_csv(results_dir / "real_version_trajectory.csv", encoding="utf-8-sig", index=False)
    stages_df.to_csv(results_dir / "chronological_size_trajectory.csv", encoding="utf-8-sig", index=False)
    steps_df.to_csv(results_dir / "chronological_step_summary.csv", encoding="utf-8-sig", index=False)
    paired_df.to_csv(results_dir / "paired_uncertainty_and_recalibration.csv", encoding="utf-8-sig", index=False)

    save_json(
        results_dir / "run_summary.json",
        {
            "status": "complete",
            "identity": identity,
            "cohort_counts_by_release": inv_summary["cohort_counts_by_release"],
            "unified_partition": inv_summary["unified_partition"],
            "task_count": len(index),
            "official_release_points": len(releases_df),
            "chronological_stages": len(stages_df),
            "full_fit_stages": int(stages_df.status.eq("fit").sum()),
            "partial_missing_class_stages": int(stages_df.status.eq("partial_missing_class").sum()),
            "missing_training_class_stages": int(stages_df.status.eq("missing_training_class").sum()),
            "stages_at_20230612": int(stages_df.verified_cut.eq("20230612").sum()),
            "split_stages": int(stages_df.boundary_group_split.sum()),
            "secondary_metrics": (
                "Classification: AUROC, AUPRC, Log-loss, CITL, E/O, 10-bin quantile ECE, release calibration slope, "
                "and closed-form Bayes prior-shift recalibration; Regression: R^2, RMSE, calibration ME, calibration slope, "
                "MedAE, NMAE, and aggregate mean-shift alignment; M4: 1,000 paired evaluation-participant bootstrap CIs"
            ),
            "followup_rule": "Unified scheduled visit filter (V[0-9]+) within 21-27 months across all longitudinal tasks",
            "partition_rule": "Single stratified 30% SHA-256 reservation by (reference_cohort, earliest_contiguous_cut)",
            "time_rule": "Earliest contiguous matching official extract through August 2026 (age tolerance 0.01 yr; exact elsewhere)",
            "tie_handling": f"{ROUNDS} nested within-snapshot permutations; earlier snapshots fixed",
            "bootstrap_rule": f"{ROUNDS} paired evaluation-participant resamples (stratified by class for classification, IID for regression; purpose=3)",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "pilot": receipt,
        },
    )
    print(
        f"Complete: {len(index)} tasks; {len(releases_df)} official release points; "
        f"{len(stages_df)} chronological stages; {len(paired_df)} paired bootstrap rows; "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
