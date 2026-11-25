"""Generate publication tables and trajectory figures from unified aggregate outputs."""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent if _HERE.name == "code" else _HERE
DEFAULT_RESULTS = (
    PROJECT / "manuscript" / "results" / "unified_dual_trajectories"
    if _HERE.name == "code"
    else PROJECT / "outputs" / "results"
)
DEFAULT_FIGURES = (
    PROJECT / "manuscript" / "figures"
    if _HERE.name == "code"
    else PROJECT / "outputs" / "figures"
)

LOCAL_DISPLAY_MAP = _HERE / "task_display_labels.json"
_TASK_ID_TOKEN = re.compile(r"\b[A-Z]\d{1,2}\b")


def display_task_text(text: str, labels: dict[str, str] | None) -> str:
    if labels is None:
        return text
    return _TASK_ID_TOKEN.sub(lambda match: labels.get(match[0], match[0]), text)


def order_for_display(table: pd.DataFrame, labels: dict[str, str] | None) -> pd.DataFrame:
    if labels is None:
        return table
    return table.set_index("task").loc[list(labels)].reset_index()


CATALOG_NAMES = {
    "T1": "Cognitive state (normal vs MCI or dementia); PD with normal cognition at baseline",
    "T2": "MoCA",
    "T3": "MDS-UPDRS II",
    "T4": "MDS-UPDRS III",
    "C01": "Recorded PD vs HC enrollment",
    "C02": "Recorded prodromal vs HC enrollment",
    "C03": "Recorded PD vs prodromal enrollment",
    "C04": "Recorded PD vs prodromal vs HC enrollment",
    "C05": "Recorded PD vs SWEDD enrollment",
    "K01": "Normal cognition vs MCI within PD",
    "K06": "SDMT",
    "K07": "Animal fluency",
    "K08": "HVLT delayed recall",
    "K09": "TMT-B",
    "F01": "MDS-UPDRS I",
    "F02": "MDS-UPDRS IV",
    "F03": "Schwab--England ADL score",
    "F04": r"H\&Y stage $\geq 3$ among those below stage 3 at baseline",
    "F05": r"Schwab--England ADL $<80$ among those scoring $\geq 80$ at baseline",
    "F06": "Any progression milestone among those with none at baseline",
    "F08": "Walking or balance milestone among those with neither at baseline",
    "F10": "PD medication or DBS among those untreated at baseline",
    "N01": "SCOPA-AUT",
    "N02": "GDS",
    "N03": "ESS",
    "N04": "STAI trait anxiety",
    "N05": "RBDSQ",
    "N07": "UPSIT olfactory score within PD",
    "S01": "Tremor-dominant vs other motor subtype within PD",
    "S02": "Tremor-dominant vs other motor subtype within PD",
    "S03": "Sporadic-type vs genetic enrollment label within PD",
    "S04": "Sporadic-type vs LRRK2 vs GBA enrollment label within PD",
    "S06": "PSG-confirmed RBD vs hyposmia without RBD in prodromal participants",
    "S09": "Recorded RBD vs hyposmia enrollment label in prodromal participants",
    "B03": "Serum NfL within PD",
    "B04": "MoCA within PD with measured baseline serum NfL",
    "B06": "Bilateral striatal DaT binding ratio (MIAKAT) within PD",
    "P01": "MoCA in prodromal participants",
    "P02": "MDS-UPDRS III in prodromal participants",
}

SHORT_LABELS = {
    "T1": "T1  2y cognitive state",
    "T2": "T2  2y MoCA",
    "K01": "K01  BL normal vs MCI",
    "K06": "K06  2y SDMT",
    "K07": "K07  2y animal fluency",
    "K08": "K08  2y HVLT recall",
    "K09": "K09  2y TMT-B",
    "T3": "T3  2y MDS-UPDRS II",
    "T4": "T4  2y MDS-UPDRS III",
    "F01": "F01  2y MDS-UPDRS I",
    "F02": "F02  2y MDS-UPDRS IV",
    "F03": "F03  2y Schwab\u2013England",
    "F04": "F04  2y H&Y \u22653",
    "F05": "F05  2y ADL <80",
    "F06": "F06  2y any milestone",
    "F08": "F08  2y walk/balance",
    "F10": "F10  2y med or DBS",
    "C01": "C01  PD vs HC",
    "C02": "C02  Prodromal vs HC",
    "C03": "C03  PD vs prodromal",
    "C04": "C04  PD/Prodromal/HC",
    "C05": "C05  PD vs SWEDD",
    "N01": "N01  2y SCOPA-AUT",
    "N02": "N02  2y GDS",
    "N03": "N03  2y ESS",
    "N04": "N04  2y STAI anxiety",
    "N05": "N05  2y RBDSQ",
    "N07": "N07  BL UPSIT",
    "S01": "S01  BL tremor subtype",
    "S02": "S02  2y tremor subtype",
    "S03": "S03  Sporadic vs genetic",
    "S04": "S04  Sporadic/LRRK2/GBA",
    "S06": "S06  PSG-RBD vs hyposmia",
    "S09": "S09  RBD vs hyposmia",
    "B03": "B03  BL serum NfL",
    "B04": "B04  2y MoCA (NfL sub.)",
    "B06": "B06  BL striatal DaT",
    "P01": "P01  2y MoCA (prod.)",
    "P02": "P02  2y MDS-UPDRS III (prod.)",
}

SUPP_FAMILY_GROUPS: list[tuple[str, str, list[str]]] = [
    ("supp_fig_s1.pdf", "Cognition (7 tasks)", ["T1", "T2", "K01", "K06", "K07", "K08", "K09"]),
    ("supp_fig_s2.pdf", "Motor and function: continuous scores (5 tasks)", ["T3", "T4", "F01", "F02", "F03"]),
    ("supp_fig_s3.pdf", "Motor and function: binary milestones and treatment (5 tasks)", ["F04", "F05", "F06", "F08", "F10"]),
    ("supp_fig_s4.pdf", "Enrollment groups (5 tasks)", ["C01", "C02", "C03", "C04", "C05"]),
    ("supp_fig_s5.pdf", "Non-motor symptoms (6 tasks)", ["N01", "N02", "N03", "N04", "N05", "N07"]),
    ("supp_fig_s6.pdf", "Clinical and genetic subtypes (6 tasks)", ["S01", "S02", "S03", "S04", "S06", "S09"]),
    ("supp_fig_s7.pdf", "Biomarkers and prodromal outcomes (5 tasks)", ["B03", "B04", "B06", "P01", "P02"]),
]

FAMILY_ORDER = [
    "Cognition",
    "Motor and function",
    "Enrollment groups",
    "Non-motor symptoms",
    "Subtypes",
    "Biomarkers",
    "Prodromal outcomes",
]

FAMILY_PREDICTOR_NOTE = {
    "Cognition": "Age+base (6); Core (1)",
    "Motor and function": "Age+base (5); Core+II (5)",
    "Enrollment groups": "Core (5)",
    "Non-motor symptoms": "Age+base (5); Core (1)",
    "Subtypes": "Core (5); Core+II/III (1)",
    "Biomarkers": "Core (2); Age+MoCA+NfL (1)",
    "Prodromal outcomes": "Age+base (2)",
}

CORE_FEATURES = ["age", "SEX", "EDUCYRS", "gds", "ess", "scopa"]
PREDICTOR_OVERRIDES = {
    "T1": "Age, MoCA",
    "T2": "Age, MoCA",
    "T3": "Age, MDS-UPDRS II",
    "T4": "Age, MDS-UPDRS III",
    "B04": "Age, MoCA, serum NfL",
}


def load_inputs(results_dir: Path):
    versions = pd.read_csv(results_dir / "real_version_trajectory.csv", dtype={"version": str})
    size = pd.read_csv(results_dir / "chronological_size_trajectory.csv", dtype={"verified_cut": str})
    tasks = pd.read_csv(results_dir / "task_index.csv", dtype={"first_version": str, "latest_version": str})
    assert versions.set_index(["task", "version"]).index.is_unique
    assert size.set_index(["task", "training_n"]).index.is_unique
    assert tasks.set_index("task").versions.eq(versions.groupby("task").size()).all()
    return tasks, versions, size


def load_paired(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "paired_uncertainty_and_recalibration.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, dtype={"first_version": str, "latest_version": str})
    assert df.set_index("task").index.is_unique
    return df


def pair_trajectories(tasks: pd.DataFrame, versions: pd.DataFrame, size: pd.DataFrame):
    for task in tasks.itertuples():
        v = versions.loc[versions.task.eq(task.task)].sort_values("version")
        n = size.loc[size.task.eq(task.task)].sort_values("training_n")
        assert v.evaluation_n.eq(task.evaluation_n).all() and n.evaluation_n.eq(task.evaluation_n).all()
        assert v.primary_metric.eq(task.primary_metric).all() and n.primary_metric.eq(task.primary_metric).all()
        assert v.version.iloc[-1] == "20260810"
        assert n.training_n.iloc[-1] == v.training_n.iloc[-1]
        assert abs(n.observed_error.iloc[-1] - v.observed_release_error.iloc[-1]) < 1e-9
        assert n.verified_cut.is_monotonic_increasing
        yield task, v, n


def _col_val(df: pd.DataFrame, col: str, idx: int) -> float:
    if col not in df.columns or len(df) == 0:
        return np.nan
    val = df[col].iloc[idx]
    return float(val) if pd.notna(val) else np.nan


def describe_trajectories(tasks: pd.DataFrame, versions: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    has_auroc_v = "auroc" in versions.columns
    has_r2_v = "r2" in versions.columns
    has_rmse_v = "rmse" in versions.columns
    has_auroc_s = "auroc_mean" in size.columns
    has_r2_s = "r2_mean" in size.columns
    has_rmse_s = "rmse_mean" in size.columns
    has_adj_v = "prior_adj_error" in versions.columns

    records = []
    for row, v, n in pair_trajectories(tasks, versions, size):
        errors = v.observed_release_error.to_numpy()
        fitted = n.loc[n.valid_rounds > 0]
        trajectory = fitted.observed_error.to_numpy()
        sizes = fitted.training_n.to_numpy()
        has_multi = len(trajectory) >= 2
        consecutive = np.diff(trajectory) if has_multi else np.array([])
        relative = 100.0 * np.ptp(errors) / errors[0]
        first_to_last_pct = 100.0 * (errors[-1] - errors[0]) / errors[0]
        small_to_full_pct = 100.0 * (trajectory[-1] - trajectory[0]) / trajectory[0] if has_multi else np.nan
        last_step_pct = 100.0 * (trajectory[-1] - trajectory[-2]) / trajectory[-2] if has_multi else np.nan
        plateau = next(
            (int(sizes[i]) for i in range(len(sizes) - 2) if 100.0 * np.ptp(trajectory[i:]) / trajectory[i] <= 2.0),
            None,
        )

        if has_adj_v:
            adj_arr = v.prior_adj_error.to_numpy(dtype=float)
            first_adj = float(adj_arr[0])
            last_adj = float(adj_arr[-1])
            adj_spread = float(100.0 * np.ptp(adj_arr) / adj_arr[0])
            adj_f2l_pct = float(100.0 * (adj_arr[-1] - adj_arr[0]) / adj_arr[0])
            raw_delta = float(errors[-1] - errors[0])
            adj_delta = float(adj_arr[-1] - adj_arr[0])
            prior_expl = float(100.0 * (raw_delta - adj_delta) / raw_delta) if raw_delta > 1e-12 else np.nan
        else:
            first_adj = last_adj = adj_spread = adj_f2l_pct = prior_expl = np.nan

        is_cls = row.primary_metric != "mae"
        first_auroc = float(v.auroc.iloc[0]) if (is_cls and has_auroc_v) else np.nan
        last_auroc = float(v.auroc.iloc[-1]) if (is_cls and has_auroc_v) else np.nan
        small_auroc = float(fitted.auroc_mean.iloc[0]) if (is_cls and has_multi and has_auroc_s) else np.nan
        small_auroc_p025 = float(fitted.auroc_p025.iloc[0]) if (is_cls and has_multi and has_auroc_s) else np.nan
        small_auroc_p975 = float(fitted.auroc_p975.iloc[0]) if (is_cls and has_multi and has_auroc_s) else np.nan
        full_auroc = float(fitted.auroc_mean.iloc[-1]) if (is_cls and has_auroc_s) else last_auroc

        first_r2 = float(v.r2.iloc[0]) if ((not is_cls) and has_r2_v) else np.nan
        last_r2 = float(v.r2.iloc[-1]) if ((not is_cls) and has_r2_v) else np.nan
        small_r2 = float(fitted.r2_mean.iloc[0]) if ((not is_cls) and has_multi and has_r2_s) else np.nan
        small_r2_p025 = float(fitted.r2_p025.iloc[0]) if ((not is_cls) and has_multi and has_r2_s) else np.nan
        small_r2_p975 = float(fitted.r2_p975.iloc[0]) if ((not is_cls) and has_multi and has_r2_s) else np.nan
        full_r2 = float(fitted.r2_mean.iloc[-1]) if ((not is_cls) and has_r2_s) else last_r2

        first_rmse = float(v.rmse.iloc[0]) if ((not is_cls) and has_rmse_v) else np.nan
        last_rmse = float(v.rmse.iloc[-1]) if ((not is_cls) and has_rmse_v) else np.nan
        small_rmse = float(fitted.rmse_mean.iloc[0]) if ((not is_cls) and has_multi and has_rmse_s) else np.nan
        small_rmse_p025 = float(fitted.rmse_p025.iloc[0]) if ((not is_cls) and has_multi and has_rmse_s) else np.nan
        small_rmse_p975 = float(fitted.rmse_p975.iloc[0]) if ((not is_cls) and has_multi and has_rmse_s) else np.nan
        full_rmse = float(fitted.rmse_mean.iloc[-1]) if ((not is_cls) and has_rmse_s) else last_rmse

        records.append(
            {
                "task": row.task,
                "family": row.family,
                "timing": row.timing,
                "versions": int(row.versions),
                "evaluation_n": int(row.evaluation_n),
                "first_dev_n": int(v.training_n.iloc[0]),
                "last_dev_n": int(v.training_n.iloc[-1]),
                "metric": row.primary_metric,
                "first_cut": float(errors[0]),
                "last_cut": float(errors[-1]),
                "release_spread": float(relative),
                "first_to_last_pct": float(first_to_last_pct),
                "release_pattern": "Narrow" if relative <= 3.0 else ("Wide" if relative >= 5.0 else "Intermediate"),
                "first_adj_cut": first_adj,
                "last_adj_cut": last_adj,
                "adj_release_spread": adj_spread,
                "adj_first_to_last_pct": adj_f2l_pct,
                "adj_release_pattern": (
                    ("Narrow" if adj_spread <= 3.0 else ("Wide" if adj_spread >= 5.0 else "Intermediate"))
                    if np.isfinite(adj_spread)
                    else ""
                ),
                "prior_explained_pct": prior_expl,
                "estimable_tiers": len(fitted),
                "small_n": int(sizes[0]) if has_multi else None,
                "small_error": float(trajectory[0]) if has_multi else np.nan,
                "small_p025": float(fitted.error_p025.iloc[0]) if has_multi else np.nan,
                "small_p975": float(fitted.error_p975.iloc[0]) if has_multi else np.nan,
                "large_n": int(sizes[-1]),
                "large_error": float(trajectory[-1]),
                "small_to_full_pct": float(small_to_full_pct) if np.isfinite(small_to_full_pct) else np.nan,
                "last_step_pct": float(last_step_pct) if np.isfinite(last_step_pct) else np.nan,
                "direction": (
                    "Declining"
                    if (has_multi and (consecutive < 0).all())
                    else ("Single tier" if not has_multi else "Reversals")
                ),
                "narrow_from": plateau,
                "first_auroc": first_auroc,
                "last_auroc": last_auroc,
                "small_auroc": small_auroc,
                "small_auroc_p025": small_auroc_p025,
                "small_auroc_p975": small_auroc_p975,
                "full_auroc": full_auroc,
                "first_r2": first_r2,
                "last_r2": last_r2,
                "small_r2": small_r2,
                "small_r2_p025": small_r2_p025,
                "small_r2_p975": small_r2_p975,
                "full_r2": full_r2,
                "first_rmse": first_rmse,
                "last_rmse": last_rmse,
                "small_rmse": small_rmse,
                "small_rmse_p025": small_rmse_p025,
                "small_rmse_p975": small_rmse_p975,
                "full_rmse": full_rmse,
                # M1-M3 diagnostic endpoints
                "first_auprc": _col_val(v, "auprc", 0) if is_cls else np.nan,
                "last_auprc": _col_val(v, "auprc", -1) if is_cls else np.nan,
                "last_auprc_lift": _col_val(v, "auprc_lift", -1) if is_cls else np.nan,
                "small_auprc": _col_val(fitted, "auprc_mean", 0) if (is_cls and has_multi) else np.nan,
                "first_logloss": _col_val(v, "log_loss", 0) if is_cls else np.nan,
                "last_logloss": _col_val(v, "log_loss", -1) if is_cls else np.nan,
                "small_logloss": _col_val(fitted, "log_loss_mean", 0) if (is_cls and has_multi) else np.nan,
                "first_adj_logloss": _col_val(v, "prior_adj_log_loss", 0) if is_cls else np.nan,
                "last_adj_logloss": _col_val(v, "prior_adj_log_loss", -1) if is_cls else np.nan,
                "first_citl": _col_val(v, "citl", 0) if is_cls else np.nan,
                "last_citl": _col_val(v, "citl", -1) if is_cls else np.nan,
                "first_macro_abs_citl": _col_val(v, "macro_abs_citl", 0) if is_cls else np.nan,
                "last_macro_abs_citl": _col_val(v, "macro_abs_citl", -1) if is_cls else np.nan,
                "first_eo": _col_val(v, "eo_ratio", 0) if is_cls else np.nan,
                "last_eo": _col_val(v, "eo_ratio", -1) if is_cls else np.nan,
                "first_ece": _col_val(v, "ece", 0) if is_cls else np.nan,
                "last_ece": _col_val(v, "ece", -1) if is_cls else np.nan,
                "first_cal_slope": _col_val(v, "cal_slope", 0),
                "last_cal_slope": _col_val(v, "cal_slope", -1),
                "first_me": _col_val(v, "cal_me", 0) if (not is_cls) else np.nan,
                "last_me": _col_val(v, "cal_me", -1) if (not is_cls) else np.nan,
                "first_medae": _col_val(v, "medae", 0) if (not is_cls) else np.nan,
                "last_medae": _col_val(v, "medae", -1) if (not is_cls) else np.nan,
                "first_nmae": _col_val(v, "nmae", 0) if (not is_cls) else np.nan,
                "last_nmae": _col_val(v, "nmae", -1) if (not is_cls) else np.nan,
                "last_adj_r2": _col_val(v, "prior_adj_r2", -1) if (not is_cls) else np.nan,
                "last_adj_rmse": _col_val(v, "prior_adj_rmse", -1) if (not is_cls) else np.nan,
            }
        )
    data = pd.DataFrame(records)
    assert len(data) == len(tasks)
    assert (data.release_pattern.value_counts().sum()) == len(tasks)
    assert (data.direction.value_counts().sum()) == len(tasks)
    return data


def _fmt_range(lo: int, hi: int) -> str:
    return f"{lo:,}" if lo == hi else f"{lo:,}--{hi:,}"


def _fmt_signed(val: float, digits: int = 3) -> str:
    if not np.isfinite(val):
        return "---"
    if abs(val) < 0.5 * (10.0 ** (-digits)):
        return f"{0.0:.{digits}f}"
    if val < 0:
        return f"$-${abs(val):.{digits}f}"
    return f"{val:.{digits}f}"


def _fmt_signed_delta(val: float, digits: int = 3) -> str:
    if not np.isfinite(val):
        return "---"
    if abs(val) < 0.5 * (10.0 ** (-digits)):
        return f"{0.0:.{digits}f}"
    if val < 0:
        return f"$-${abs(val):.{digits}f}"
    return f"$+${val:.{digits}f}"


def make_design_table(
    data: pd.DataFrame, results_dir: Path, task_labels: dict[str, str] | None = None
) -> None:
    """Table 1: Task-family design overview and sample support across the 11 curated releases."""
    header = (
        r"\begingroup\setlength{\tabcolsep}{3pt}%" "\n"
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xcccrr@{}}" "\n"
        r"\toprule" "\n"
        r"\textbf{Task family} & \textbf{Tasks} & \textbf{Horizon} & "
        r"\textbf{Cuts} & \textbf{Dev.\ $n$} & \textbf{Eval.\ $n$}" "\\\\\n"
        r" & \textbf{(Cls/Reg)} & \textbf{(BL/2y)} & & \textbf{(first $\to$ last)} & " "\\\\\n"
        r"\midrule"
    )
    rows = []
    present_families = [f for f in FAMILY_ORDER if f in set(data.family)]
    for extra in data.family.unique():
        if extra not in present_families:
            present_families.append(extra)
    for fam in present_families:
        g = data.loc[data.family.eq(fam)]
        n_cls = int((g.metric != "mae").sum())
        n_reg = int((g.metric == "mae").sum())
        n_bl = int(g.timing.eq("BL").sum())
        n_m24 = int(g.timing.eq("M24").sum())
        cuts_str = _fmt_range(int(g.versions.min()), int(g.versions.max()))
        dev_str = f"{_fmt_range(int(g.first_dev_n.min()), int(g.first_dev_n.max()))} $\\to$ {_fmt_range(int(g.last_dev_n.min()), int(g.last_dev_n.max()))}"
        eval_str = _fmt_range(int(g.evaluation_n.min()), int(g.evaluation_n.max()))
        fam_display = fam
        if task_labels is not None and len(g):
            disp_ids = [task_labels[t] for t in g.task if t in task_labels]
            if disp_ids:
                id_span = disp_ids[0] if len(disp_ids) == 1 else f"{disp_ids[0]}--{disp_ids[-1]}"
                fam_display = f"{fam} ({id_span})"
        rows.append(
            f"{fam_display} & {len(g)} ({n_cls}/{n_reg}) & {n_bl} / {n_m24} & {cuts_str} & {dev_str} & {eval_str} " + r"\\"
        )

    tot_cls = int((data.metric != "mae").sum())
    tot_reg = int((data.metric == "mae").sum())
    tot_bl = int(data.timing.eq("BL").sum())
    tot_m24 = int(data.timing.eq("M24").sum())
    tot_cuts = _fmt_range(int(data.versions.min()), int(data.versions.max()))
    tot_dev = f"{_fmt_range(int(data.first_dev_n.min()), int(data.first_dev_n.max()))} $\\to$ {_fmt_range(int(data.last_dev_n.min()), int(data.last_dev_n.max()))}"
    tot_eval = _fmt_range(int(data.evaluation_n.min()), int(data.evaluation_n.max()))
    rows.append(
        f"\\midrule Total / range & {len(data)} ({tot_cls}/{tot_reg}) & {tot_bl} / {tot_m24} & "
        f"{tot_cuts} & {tot_dev} & {tot_eval} " + r"\\"
    )
    (results_dir / "paper_design_table.tex").write_text(
        "\n".join([header, *rows, r"\bottomrule", r"\end{tabularx}%", r"\endgroup", ""]), encoding="utf-8"
    )


def _fmt_ci_cell(diff: float, lo: float, hi: float, digits: int = 4) -> str:
    if not (np.isfinite(diff) and np.isfinite(lo) and np.isfinite(hi)):
        return "---"
    sig = "*" if (hi < 0.0 or lo > 0.0) else ""
    d_str = _fmt_signed_delta(diff, digits)
    lo_str = _fmt_signed(lo, digits)
    hi_str = _fmt_signed(hi, digits)
    return f"{d_str}{sig} [{lo_str}, {hi_str}]"


def make_tables(
    data: pd.DataFrame,
    size: pd.DataFrame,
    results_dir: Path,
    paired: pd.DataFrame | None = None,
    task_labels: dict[str, str] | None = None,
) -> None:
    make_design_table(data, results_dir, task_labels=task_labels)

    has_adj = "adj_release_spread" in data.columns and data.adj_release_spread.notna().any()

    # Table 2 (main): Dual-trajectory patterns, raw->prior-adjusted spread, and full-pool benchmarks
    spread_col_h1 = r"\textbf{Spread \% med.}" if has_adj else r"\textbf{Spread \%}"
    spread_col_h2 = r"\textbf{(raw $\to$ adj.)}" if has_adj else r"\textbf{med.\ (range)}"
    header = (
        r"\begingroup\setlength{\tabcolsep}{2.5pt}%" "\n"
        r"\begin{tabularx}{\linewidth}{@{}Xrrrcccr@{}}" "\n"
        r"\toprule" "\n"
        r"\textbf{Task family} & \textbf{Tasks} & \textbf{Lower} & \textbf{Narrow/} & "
        + spread_col_h1
        + r" & \textbf{Decl./} & \textbf{Full} & \textbf{Full}" "\\\\\n"
        r" & & \textbf{at last} & \textbf{Wide} & "
        + spread_col_h2
        + r" & \textbf{Early} & \textbf{AUROC} & \textbf{$R^2$}" "\\\\\n"
        r"\midrule"
    )
    rows = []
    present_families = [f for f in FAMILY_ORDER if f in set(data.family)]
    for extra in data.family.unique():
        if extra not in present_families:
            present_families.append(extra)
    for fam in present_families:
        group = data.loc[data.family.eq(fam)]
        n_low = int((group.last_cut < group.first_cut).sum())
        n_nar = int(group.release_pattern.eq("Narrow").sum())
        n_wid = int(group.release_pattern.eq("Wide").sum())
        sp_med = float(group.release_spread.median())
        if has_adj:
            adj_med = float(group.adj_release_spread.median())
            sp_str = f"{sp_med:.2f} $\\to$ {adj_med:.2f}"
        else:
            sp_lo, sp_hi = float(group.release_spread.min()), float(group.release_spread.max())
            sp_str = f"{sp_med:.2f} ({sp_lo:.2f}--{sp_hi:.2f})"
        n_dec = int(group.direction.eq("Declining").sum())
        n_ear = int(group.narrow_from.notna().sum())
        aucs = group.full_auroc.dropna()
        r2s = group.full_r2.dropna()
        auc_str = f"{aucs.min():.3f}--{aucs.max():.3f}" if len(aucs) > 1 else (f"{aucs.iloc[0]:.3f}" if len(aucs) == 1 else "---")
        r2_str = (
            f"{_fmt_signed(float(r2s.min()), 3)}--{_fmt_signed(float(r2s.max()), 3)}"
            if len(r2s) > 1
            else (_fmt_signed(float(r2s.iloc[0]), 3) if len(r2s) == 1 else "---")
        )
        rows.append(
            f"{fam} & {len(group)} & {n_low} & {n_nar} / {n_wid} & {sp_str} & {n_dec} / {n_ear} & {auc_str} & {r2_str} "
            + r"\\"
        )

    tot_low = int((data.last_cut < data.first_cut).sum())
    tot_nar = int(data.release_pattern.eq("Narrow").sum())
    tot_wid = int(data.release_pattern.eq("Wide").sum())
    if has_adj:
        tot_sp = f"{float(data.release_spread.median()):.2f} $\\to$ {float(data.adj_release_spread.median()):.2f}"
    else:
        tot_sp = f"{float(data.release_spread.median()):.2f} ({float(data.release_spread.min()):.2f}--{float(data.release_spread.max()):.2f})"
    tot_dec = int(data.direction.eq("Declining").sum())
    tot_ear = int(data.narrow_from.notna().sum())
    all_aucs = data.full_auroc.dropna()
    all_r2s = data.full_r2.dropna()
    tot_auc = f"{all_aucs.min():.3f}--{all_aucs.max():.3f}" if len(all_aucs) else "---"
    tot_r2 = (
        f"{_fmt_signed(float(all_r2s.min()), 3)}--{_fmt_signed(float(all_r2s.max()), 3)}"
        if len(all_r2s)
        else "---"
    )
    rows.append(
        f"\\midrule Total / range & {len(data)} & {tot_low} & {tot_nar} / {tot_wid} & {tot_sp} & "
        f"{tot_dec} / {tot_ear} & {tot_auc} & {tot_r2} " + r"\\"
    )
    (results_dir / "paper_family_table.tex").write_text(
        "\n".join([header, *rows, r"\bottomrule", r"\end{tabularx}%", r"\endgroup", ""]), encoding="utf-8"
    )

    # Table S2: Primary error by task across both trajectories
    c05 = size.loc[size.task.eq("C05")].sort_values("training_n")
    c05_miss = c05.loc[c05.status.eq("missing_training_class")]
    c05_note = ""
    if len(c05_miss):
        c05_note = (
            f" In C05, the {len(c05_miss)} smaller tiers "
            f"($n={int(c05_miss.training_n.iloc[0]):,}$--${int(c05_miss.training_n.iloc[-1]):,}$) "
            f"fell in June 2023 before SWEDD participants entered the curated archive and contained no second class; "
            f"only the full tier ($n={int(c05.training_n.iloc[-1]):,}$) was estimable."
        )

    spec = (
        r"\begingroup\renewcommand{\ManuscriptTableFontSize}{8.0}\renewcommand{\ManuscriptTableBaseline}{9.6}"
        r"\setlength{\tabcolsep}{2.5pt}\setlength{\LTcapwidth}{\linewidth}%" "\n"
        r"\begin{longtable}{@{}lcrrrrrrr@{}}"
    )
    caption = (
        r"\caption{\textbf{Primary error and development sample size by task on the two trajectories.} "
        r"Dev.\ $n$ reports all eligible non-reserved development records at the first and latest included extracts (1st $\to$ last; "
        r"fixed evaluation counts appear in Table~\ref{tab:scatalog}). First and last refer to the "
        r"available official release dates. Spread is the full release range relative to first-release error. "
        r"Small mean and Small 95\% range report the mean and 2.5th--97.5th percentile across 1,000 "
        r"within-snapshot tie permutations at the earliest estimable chronological tier (for tasks with at "
        r"least two estimable tiers)." + c05_note +
        r" Narrow from is the earliest tier with at least two subsequent "
        r"mean estimates within the descriptive 2\% range rule. Regression MAE values are shown to four "
        r"decimal places (three in the range column) and classification Brier scores to five (four in the "
        r"range column).}\label{tab:sresults}\\"
    )
    heading = (
        r"\toprule" "\n"
        r"\textbf{Task} & \textbf{Dev.\ $n$} & \textbf{First cut} & \textbf{Last cut} & \textbf{Spread (\%)} & "
        r"\textbf{Small mean} & \textbf{Small 95\% range} & \textbf{Full} & \textbf{Narrow from} \\" "\n"
        r"\midrule"
    )
    rows = []
    for item in data.itertuples():
        digits = 4 if item.metric == "mae" else 5
        rdigits = 3 if item.metric == "mae" else 4
        dev_str = f"{int(item.first_dev_n):,}$\\to${int(item.last_dev_n):,}"
        narrow = f"{int(item.narrow_from):,}" if pd.notna(item.narrow_from) else "---"
        if pd.notna(item.small_error):
            small_str = f"{item.small_error:.{digits}f}"
            range_str = f"{item.small_p025:.{rdigits}f}--{item.small_p975:.{rdigits}f}"
        else:
            small_str = "---"
            range_str = "---"
        rows.append(
            f"{item.task} & {dev_str} & {item.first_cut:.{digits}f} & {item.last_cut:.{digits}f} & "
            f"{item.release_spread:.2f} & {small_str} & {range_str} & "
            f"{item.large_error:.{digits}f} & {narrow} " + r"\\"
        )
    cont_s2 = r"\multicolumn{9}{@{}l}{\fontsize{10pt}{12pt}\selectfont\textbf{Table~\thetable\ -- continued from previous page}}\\[4pt]"
    text = [spec, caption, heading, r"\endfirsthead", cont_s2, heading, r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}%", r"\endgroup", ""]
    (results_dir / "paper_task_results.tex").write_text(
        display_task_text("\n".join(text), task_labels), encoding="utf-8"
    )

    # Table S3: Classification discrimination & proper scoring (AUROC, AUPRC, Log-loss)
    cls_data = data.loc[data.metric != "mae"]
    has_ap = "last_auprc" in cls_data.columns and cls_data.last_auprc.notna().any()
    if len(cls_data) and cls_data.full_auroc.notna().any():
        cont_s3 = r"\multicolumn{8}{@{}l}{\fontsize{10pt}{12pt}\selectfont\textbf{Table~\thetable\ -- continued from previous page}}\\[4pt]"
        if has_ap:
            spec_s3 = r"\begingroup\footnotesize\setlength{\tabcolsep}{5pt}\setlength{\LTcapwidth}{\linewidth}%" "\n" r"\begin{longtable}{@{}llcccccc@{}}"
            cap_s3 = (
                r"\caption{\textbf{Classification discrimination (AUROC, AUPRC) and log-loss across official releases and chronological growth.} "
                r"Binary tasks report ROC AUROC and exact average precision (AUPRC); three-class tasks (C04 and S04) report macro-averaged one-vs-rest (OvR) "
                r"AUROC and AUPRC. Lift is August 2026 AUPRC minus evaluation baseline prevalence (or $1/3$ for three-class tasks). "
                r"Small $\to$ Full AUROC shows chronological permutation mean at the earliest estimable tier to full pool (C05 had only the full tier estimable). "
                r"Adj.\ LL is August 2026 log-loss after zero-leakage closed-form Bayes prior-ratio recalibration.}\label{tab:s_auroc}\\"
            )
            head_s3 = (
                r"\toprule" "\n"
                r"\textbf{Task} & \textbf{Type} & \textbf{AUROC (1st $\to$ Last)} & \textbf{AUROC (Small $\to$ Full)} & "
                r"\textbf{AUPRC (1st $\to$ Last)} & \textbf{Lift} & \textbf{LL (1st $\to$ Last)} & \textbf{Adj.\ LL} \\" "\n"
                r"\midrule"
            )
            rows_s3 = []
            for item in cls_data.itertuples():
                kind = "3-cls" if item.metric == "multiclass_brier" else "Binary"
                auc_rel = f"{item.first_auroc:.3f} $\\to$ {item.last_auroc:.3f}"
                auc_ch = (
                    f"{item.small_auroc:.3f} $\\to$ {item.full_auroc:.3f}"
                    if pd.notna(item.small_auroc)
                    else f"--- $\\to$ {item.full_auroc:.3f}"
                )
                ap_rel = f"{item.first_auprc:.3f} $\\to$ {item.last_auprc:.3f}"
                lift_s = _fmt_signed_delta(item.last_auprc_lift, 3)
                ll_rel = f"{item.first_logloss:.3f} $\\to$ {item.last_logloss:.3f}"
                adj_ll = f"{item.last_adj_logloss:.3f}"
                rows_s3.append(
                    f"{item.task} & {kind} & {auc_rel} & {auc_ch} & {ap_rel} & {lift_s} & {ll_rel} & {adj_ll} " + r"\\"
                )
            txt_s3 = [spec_s3, cap_s3, head_s3, r"\endfirsthead", cont_s3, head_s3, r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot", *rows_s3, r"\end{longtable}%", r"\endgroup", ""]
        else:
            spec_s3 = r"\begin{longtable}{@{}llrrrrrr@{}}"
            cap_s3 = (
                r"\caption{\textbf{Classification discrimination (AUROC) across official releases and chronological training sizes.} "
                r"Binary tasks report ROC AUROC and three-class tasks (C04 and S04) report macro-averaged one-vs-rest (OvR) AUROC.}\label{tab:s_auroc}\\"
            )
            head_s3 = (
                r"\toprule" "\n"
                r"\textbf{Task} & \textbf{Type} & \textbf{First cut} & \textbf{Last cut} & "
                r"\textbf{$\Delta$AUROC} & \textbf{Small mean} & \textbf{Small 95\% range} & \textbf{Full} \\" "\n"
                r"\midrule"
            )
            rows_s3 = []
            for item in cls_data.itertuples():
                kind = "3-class OvR" if item.metric == "multiclass_brier" else "Binary"
                d_auc = item.last_auroc - item.first_auroc
                sm_str = f"{item.small_auroc:.4f}" if pd.notna(item.small_auroc) else "---"
                rng_str = f"{item.small_auroc_p025:.3f}--{item.small_auroc_p975:.3f}" if pd.notna(item.small_auroc) else "---"
                rows_s3.append(
                    f"{item.task} & {kind} & {item.first_auroc:.4f} & {item.last_auroc:.4f} & "
                    f"{_fmt_signed_delta(d_auc, 4)} & {sm_str} & {rng_str} & {item.full_auroc:.4f} " + r"\\"
                )
            txt_s3 = [spec_s3, cap_s3, head_s3, r"\endfirsthead", cont_s3, head_s3, r"\endhead", r"\bottomrule", r"\endlastfoot", *rows_s3, r"\end{longtable}", ""]
        (results_dir / "paper_secondary_classification.tex").write_text(
            display_task_text("\n".join(txt_s3), task_labels), encoding="utf-8"
        )

    # Table S4 (NEW): Classification explicit calibration & Bayes prior-shift counterfactual decomposition
    if len(cls_data) and "last_ece" in cls_data.columns and cls_data.last_ece.notna().any():
        spec_s4c = r"\begingroup\footnotesize\setlength{\tabcolsep}{5pt}\setlength{\LTcapwidth}{\linewidth}%" "\n" r"\begin{longtable}{@{}lccccrrc@{}}"
        cap_s4c = (
            r"\caption{\textbf{Classification calibration diagnostics and closed-form Bayes prior-shift counterfactual across official releases.} "
            r"For binary tasks, CITL is fixed-slope-1 calibration-in-the-large (negative values denote over-prediction of the coded positive class), "
            r"E/O is mean predicted probability divided by observed evaluation event rate, and Slope is unpenalized logistic calibration slope. "
            r"For three-class tasks (C04 and S04), CITL shows macro-averaged $|$CITL$|$ across classes (marked \dag) and ECE is classwise 10-bin quantile ECE. "
            r"Raw sp.\ and Adj.\ sp.\ compare official-release Brier spread (\%) before and after zero-leakage closed-form Bayes prior-ratio alignment.}\label{tab:s_cls_cal}\\"
        )
        head_s4c = (
            r"\toprule" "\n"
            r"\textbf{Task} & \textbf{CITL (1st $\to$ Last)} & \textbf{E/O (1st $\to$ Last)} & "
            r"\textbf{Slope (1st $\to$ Last)} & \textbf{ECE (1st $\to$ Last)} & "
            r"\textbf{Raw sp.\%} & \textbf{Adj.\ sp.\%} & \textbf{Adj.\ Brier (1st $\to$ Last)} \\" "\n"
            r"\midrule"
        )
        rows_s4c = []
        for item in cls_data.itertuples():
            if item.metric == "multiclass_brier":
                citl_s = f"{item.first_macro_abs_citl:.2f} $\\to$ {item.last_macro_abs_citl:.2f}\\dag"
                eo_s = "---"
                slp_s = "---"
            else:
                citl_s = f"{_fmt_signed_delta(item.first_citl, 2)} $\\to$ {_fmt_signed_delta(item.last_citl, 2)}"
                eo_s = f"{item.first_eo:.2f} $\\to$ {item.last_eo:.2f}"
                slp_s = f"{item.first_cal_slope:.2f} $\\to$ {item.last_cal_slope:.2f}"
            ece_s = f"{item.first_ece:.3f} $\\to$ {item.last_ece:.3f}"
            adj_b = f"{item.first_adj_cut:.4f} $\\to$ {item.last_adj_cut:.4f}"
            rows_s4c.append(
                f"{item.task} & {citl_s} & {eo_s} & {slp_s} & {ece_s} & "
                f"{item.release_spread:.2f} & {item.adj_release_spread:.2f} & {adj_b} " + r"\\"
            )
        cont_s4c = r"\multicolumn{8}{@{}l}{\fontsize{10pt}{12pt}\selectfont\textbf{Table~\thetable\ -- continued from previous page}}\\[4pt]"
        txt_s4c = [spec_s4c, cap_s4c, head_s4c, r"\endfirsthead", cont_s4c, head_s4c, r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot", *rows_s4c, r"\end{longtable}%", r"\endgroup", ""]
        (results_dir / "paper_calibration_classification.tex").write_text(
            display_task_text("\n".join(txt_s4c), task_labels), encoding="utf-8"
        )

    # Table S5 (was S4): Regression scale-free, calibration & tail-robust metrics
    reg_data = data.loc[data.metric == "mae"]
    has_reg_cal = "last_me" in reg_data.columns and reg_data.last_me.notna().any()
    if len(reg_data) and reg_data.full_r2.notna().any():
        if has_reg_cal:
            spec_s5 = r"\begingroup\footnotesize\setlength{\tabcolsep}{5pt}\setlength{\LTcapwidth}{\linewidth}%" "\n" r"\begin{longtable}{@{}lcrcrcrrr@{}}"
            cap_s5 = (
                r"\caption{\textbf{Regression explained variance ($R^2$), RMSE, calibration, tail-robust error and mean-shift alignment across official releases.} "
                r"ME is mean prediction error ($\bar{\hat y}-\bar y$, calibration-in-the-large) in original clinical units; "
                r"Slope is the August 2026 OLS regression calibration slope ($y$ on $\hat y$); MedAE is August 2026 median absolute error; "
                r"NMAE is August 2026 MAE divided by evaluation outcome SD; Adj.\ $R^2$ and Adj.\ sp.\% reflect aggregate mean-shift alignment "
                r"($\hat y + [\bar y_{\mathrm{ref}}-\bar y_{\mathrm{train}}]$).}\label{tab:s_reg_secondary}\\"
            )
            head_s5 = (
                r"\toprule" "\n"
                r"\textbf{Task} & \textbf{$R^2$ (1st $\to$ Last)} & \textbf{Adj.\ $R^2$} & "
                r"\textbf{RMSE (1st $\to$ Last)} & \textbf{ME (1st $\to$ Last)} & \textbf{Slope} & "
                r"\textbf{MedAE} & \textbf{NMAE} & \textbf{Adj.\ sp.\%} \\" "\n"
                r"\midrule"
            )
            rows_s5 = []
            for item in reg_data.itertuples():
                r2_rel = f"{_fmt_signed(item.first_r2, 3)} $\\to$ {_fmt_signed(item.last_r2, 3)}"
                adj_r2_s = _fmt_signed(item.last_adj_r2, 3)
                rmse_rel = f"{item.first_rmse:.2f} $\\to$ {item.last_rmse:.2f}"
                me_rel = f"{_fmt_signed_delta(item.first_me, 2)} $\\to$ {_fmt_signed_delta(item.last_me, 2)}"
                rows_s5.append(
                    f"{item.task} & {r2_rel} & {adj_r2_s} & {rmse_rel} & {me_rel} & "
                    f"{item.last_cal_slope:.2f} & {item.last_medae:.2f} & {item.last_nmae:.3f} & {item.adj_release_spread:.2f} " + r"\\"
                )
            cont_s5 = r"\multicolumn{9}{@{}l}{\fontsize{10pt}{12pt}\selectfont\textbf{Table~\thetable\ -- continued from previous page}}\\[4pt]"
            txt_s5 = [spec_s5, cap_s5, head_s5, r"\endfirsthead", cont_s5, head_s5, r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot", *rows_s5, r"\end{longtable}%", r"\endgroup", ""]
        else:
            spec_s5 = r"\begin{longtable}{@{}lrrrrrrr@{}}"
            cap_s5 = (
                r"\caption{\textbf{Regression variance explained ($R^2$) and root mean squared error (RMSE) across official releases "
                r"and chronological training sizes.}}\label{tab:s_reg_secondary}\\"
            )
            head_s5 = (
                r"\toprule" "\n"
                r"\textbf{Task} & \textbf{First} & \textbf{Last} & \textbf{Small} & "
                r"\textbf{Small $R^2$} & \textbf{First} & \textbf{Small} & \textbf{Full} \\" "\n"
                r" & \textbf{$R^2$} & \textbf{$R^2$} & \textbf{$R^2$ mean} & "
                r"\textbf{95\% range} & \textbf{RMSE} & \textbf{RMSE} & \textbf{RMSE} \\" "\n"
                r"\midrule"
            )
            rows_s5 = []
            for item in reg_data.itertuples():
                r2_rng = f"{_fmt_signed(item.small_r2_p025, 3)}--{_fmt_signed(item.small_r2_p975, 3)}"
                rows_s5.append(
                    f"{item.task} & {_fmt_signed(item.first_r2, 4)} & {_fmt_signed(item.last_r2, 4)} & "
                    f"{_fmt_signed(item.small_r2, 4)} & {r2_rng} & "
                    f"{item.first_rmse:.3f} & {item.small_rmse:.3f} & {item.full_rmse:.3f} " + r"\\"
                )
            cont_s5 = r"\multicolumn{8}{@{}l}{\fontsize{10pt}{12pt}\selectfont\textbf{Table~\thetable\ -- continued from previous page}}\\[4pt]"
            txt_s5 = [spec_s5, cap_s5, head_s5, r"\endfirsthead", cont_s5, head_s5, r"\endhead", r"\bottomrule", r"\endlastfoot", *rows_s5, r"\end{longtable}", ""]
        (results_dir / "paper_secondary_regression.tex").write_text(
            display_task_text("\n".join(txt_s5), task_labels), encoding="utf-8"
        )

    # Table S6 (NEW): M4 paired evaluation-participant bootstrap 95% CIs for all 39 tasks
    if paired is not None and len(paired):
        p_map = paired.set_index("task")
        spec_s6 = r"\begingroup\scriptsize\setlength{\tabcolsep}{3.5pt}\setlength{\LTcapwidth}{\linewidth}%" "\n" r"\begin{longtable}{@{}lccccc@{}}"
        cap_s6 = (
            r"\caption{\textbf{Conditional paired evaluation-participant bootstrap 95\% confidence intervals across all 39 tasks (1,000 resamples).} "
            r"Intervals are conditional on the fixed development pools and fitted models (stratified by outcome class for classification; IID for regression). "
            r"Release Raw $\Delta$ and Release Adj.\ $\Delta$ report latest minus first official-release primary error before and after zero-leakage prior/mean alignment "
            r"(negative values indicate lower error at the latest extract). Aug 2026 Raw$-$Adj reports August 2026 error reduction from closed-form alignment "
            r"(positive values indicate error reduction). Final Step $\Delta$ (full minus penultimate tier) and Earliest $\to$ Full $\Delta$ (full minus earliest estimable tier) "
            r"pair full-pool individual losses against tie-averaged individual losses on the chronological trajectory. Asterisks (*) mark 95\% intervals strictly excluding zero before rounding.}\label{tab:s_bootstrap}\\"
        )
        head_s6 = (
            r"\toprule" "\n"
            r"\textbf{Task} & \textbf{Release Raw $\Delta$ [95\% CI]} & \textbf{Release Adj.\ $\Delta$ [95\% CI]} & "
            r"\textbf{Aug 2026 Raw$-$Adj [95\% CI]} & \textbf{Final Step $\Delta$ [95\% CI]} & "
            r"\textbf{Earliest $\to$ Full $\Delta$ [95\% CI]} \\" "\n"
            r"\midrule"
        )
        rows_s6 = []
        for item in data.itertuples():
            if item.task not in p_map.index:
                continue
            pr = p_map.loc[item.task]
            dig = 3 if item.metric == "mae" else 4
            c1 = _fmt_ci_cell(float(pr.rel_err_diff), float(pr.rel_err_ci_lo), float(pr.rel_err_ci_hi), dig)
            c2 = _fmt_ci_cell(float(pr.rel_adj_err_diff), float(pr.rel_adj_err_ci_lo), float(pr.rel_adj_err_ci_hi), dig)
            c3 = _fmt_ci_cell(
                float(pr.rel_last_raw_minus_adj),
                float(pr.rel_last_raw_minus_adj_ci_lo),
                float(pr.rel_last_raw_minus_adj_ci_hi),
                dig,
            )
            c4 = _fmt_ci_cell(float(pr.step_err_diff), float(pr.step_err_ci_lo), float(pr.step_err_ci_hi), dig)
            c5 = _fmt_ci_cell(float(pr.span_err_diff), float(pr.span_err_ci_lo), float(pr.span_err_ci_hi), dig)
            rows_s6.append(f"{item.task} & {c1} & {c2} & {c3} & {c4} & {c5} " + r"\\")
        cont_s6 = r"\multicolumn{6}{@{}l}{\fontsize{10pt}{12pt}\selectfont\textbf{Table~\thetable\ -- continued from previous page}}\\[4pt]"
        txt_s6 = [spec_s6, cap_s6, head_s6, r"\endfirsthead", cont_s6, head_s6, r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot", *rows_s6, r"\end{longtable}%", r"\endgroup", ""]
        (results_dir / "paper_paired_bootstrap.tex").write_text(
            display_task_text("\n".join(txt_s6), task_labels), encoding="utf-8"
        )


def make_task_catalog(
    tasks: pd.DataFrame, results_dir: Path, task_labels: dict[str, str] | None = None
) -> None:
    f02_row = tasks.loc[tasks.task.eq("F02")].iloc[0]
    f02_dev_miss = json.loads(f02_row.dev_missingness_pct)["updrs4_score"]
    f02_eval_miss = json.loads(f02_row.eval_missingness_pct)["updrs4_score"]

    rows = []
    for row in tasks.itertuples():
        target = CATALOG_NAMES.get(row.task, row.title.removeprefix("Two-year "))
        if row.task in PREDICTOR_OVERRIDES:
            spec = PREDICTOR_OVERRIDES[row.task]
        else:
            features = json.loads(row.features)
            if features == CORE_FEATURES:
                spec = "Core"
            elif features == CORE_FEATURES + ["updrs2_score"]:
                spec = "Core, MDS-UPDRS II"
            elif features == CORE_FEATURES + ["updrs2_score", "updrs3_score"]:
                spec = "Core, MDS-UPDRS II and III"
            else:
                spec = "Age, baseline target score"
        timing = "Baseline" if row.timing == "BL" else "2 years"
        counts = json.loads(row.evaluation_classes)
        denominator = str(row.evaluation_n)
        if counts:
            denominator += " (" + "/".join(map(str, counts.values())) + ")"
        rows.append(f"{row.task} & {target} & {timing} & {spec} & {denominator} " + r"\\")

    caption = (
        r"\caption{\textbf{Task definitions, predictors and fixed evaluation sizes.}"
        r" Evaluation reports the fixed early-anchored 30\% reserved evaluation count (class counts in parentheses follow the stated class order,"
        r" or absence then presence for binary milestones); per-task development sizes from first to latest extract (349--3,529 at August 2026) appear in Table~\ref{tab:sresults}."
        r" Core denotes age, sex, years of education, GDS, ESS and SCOPA-AUT."
        r" MDS-UPDRS III outcomes and baseline predictors use the curated OFF (off-medication) or untreated score."
        f" In F02, baseline MDS-UPDRS IV is unavailable for {f02_dev_miss:.1f}\\% of development and "
        f"{f02_eval_miss:.1f}\\% of evaluation participants (untreated baseline PD).}}"
        r"\label{tab:scatalog}\\"
    )
    heading = (
        r"\toprule" "\n"
        r"\textbf{ID} & \textbf{Target and eligibility} & \textbf{Time} & \textbf{Baseline inputs} & "
        r"\textbf{Evaluation} \\" "\n"
        r"\midrule"
    )
    spec = (
        r"\begingroup\renewcommand{\ManuscriptTableFontSize}{8.5}\renewcommand{\ManuscriptTableBaseline}{10.0}"
        r"\setlength{\tabcolsep}{3.5pt}\setlength{\LTcapwidth}{\linewidth}%" "\n"
        + (r"\begin{longtable}{@{}l>{\raggedright\arraybackslash}p{.402\linewidth}lll@{}}"
           if task_labels is not None
           else r"\begin{longtable}{@{}l>{\raggedright\arraybackslash}p{.420\linewidth}lll@{}}")
    )
    cont_s1 = r"\multicolumn{5}{@{}l}{\fontsize{10pt}{12pt}\selectfont\textbf{Table~\thetable\ -- continued from previous page}}\\[4pt]"
    text = [spec, caption, heading, r"\endfirsthead", cont_s1, heading, r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}%", r"\endgroup", ""]
    (results_dir / "paper_task_catalog.tex").write_text(
        display_task_text("\n".join(text), task_labels), encoding="utf-8"
    )


def date_ticks(dates: pd.Series) -> pd.Series:
    numbers = dates.astype("int64").to_numpy()
    proposed = np.linspace(numbers[0], numbers[-1], 4)
    nearest = np.unique([np.argmin(np.abs(numbers - v)) for v in proposed])
    keep = [0]
    min_space = (numbers[-1] - numbers[0]) / 5.0
    for i in nearest[1:-1]:
        if numbers[i] - numbers[keep[-1]] >= min_space and numbers[-1] - numbers[i] >= min_space:
            keep.append(i)
    keep.append(len(numbers) - 1)
    return dates.iloc[keep]


def make_figure2_cohort_and_shift(
    tasks: pd.DataFrame,
    versions: pd.DataFrame,
    data: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    task_labels: dict[str, str] | None = None,
) -> None:
    """Figure 2: Cohort expansion, class-proportion shift, Brier-vs-AUROC dissociation, and prior-shift decomposition."""
    inv_path = results_dir / "inventory_summary.json"
    if not inv_path.is_file():
        return
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    cc = inv.get("cohort_counts_by_release")
    if not cc:
        return

    has_adj = "adj_first_to_last_pct" in data.columns and data.adj_first_to_last_pct.notna().any()
    fig = plt.figure(figsize=(8.8, 7.6))
    if has_adj:
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.46, left=0.085, right=0.975, top=0.94, bottom=0.075)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.14], hspace=0.40, wspace=0.48, left=0.085, right=0.975, top=0.94, bottom=0.08)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, :])
        ax_d = None

    navy = "#274a69"
    teal = "#176957"
    amber = "#c27d1a"
    rust = "#b24337"
    slate = "#5a6c7a"

    # --- Panel A: Cohort expansion across 11 official curated extracts ---
    cut_keys = sorted(cc.keys())
    dates = pd.to_datetime(cut_keys, format="%Y%m%d")
    series = {
        "Prodromal": ([cc[k]["Prodromal"] for k in cut_keys], teal, "o"),
        "PD": ([cc[k]["PD"] for k in cut_keys], navy, "s"),
        "HC": ([cc[k]["HC"] for k in cut_keys], amber, "^"),
        "SWEDD": ([cc[k]["SWEDD"] for k in cut_keys], slate, "D"),
    }
    for label, (vals, col, mk) in series.items():
        ax_a.plot(dates, vals, f"-{mk}", color=col, linewidth=1.55, markersize=3.4, label=label)

    swedd_date = pd.to_datetime("20240129", format="%Y%m%d")
    ax_a.axvline(swedd_date, color=slate, linestyle="--", linewidth=0.9, alpha=0.75)
    ax_a.text(swedd_date + pd.Timedelta(days=18), 2780, "SWEDD enters\n(Jan 2024)", fontsize=6.9, color=slate, va="top")

    end_x = dates[-1] + pd.Timedelta(days=42)
    ax_a.text(end_x, cc[cut_keys[-1]]["Prodromal"], "Prodromal\n2,930 (+208%)", fontsize=6.6, color=teal, va="center", weight="bold")
    ax_a.text(end_x, cc[cut_keys[-1]]["PD"], "PD\n1,681 (+49%)", fontsize=6.6, color=navy, va="center", weight="bold")
    ax_a.text(end_x, cc[cut_keys[-1]]["HC"] + 150, "HC: 336 (+28%)", fontsize=6.5, color=amber, va="center", weight="bold")
    ax_a.text(end_x, cc[cut_keys[-1]]["SWEDD"] - 10, "SWEDD: 64", fontsize=6.5, color=slate, va="center", weight="bold")

    ax_a.set_xlim(dates[0] - pd.Timedelta(days=30), dates[-1] + pd.Timedelta(days=430))
    ax_a.set_ylim(-90, 3250)
    ticks = date_ticks(pd.Series(dates))
    ax_a.set_xticks(ticks, ticks.dt.strftime("%Y-%m"))
    ax_a.set_ylabel("Eligible baseline participants", fontsize=8.3)
    ax_a.set_xlabel("Official curated extract date", fontsize=8.3)
    ax_a.set_title("A. Cohort growth across 11 extracts", loc="left", fontsize=8.9, weight="bold", color="#22313b")
    ax_a.grid(axis="y", color="#e0eaef", linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.tick_params(labelsize=7.5)

    # --- Panel B: Class-proportion shift (First dev vs Fixed eval vs Latest dev) ---
    shift_specs = [
        ("S09", "RBD label", "S09  RBD %"),
        ("C03", "PD", "C03  PD %"),
        ("C04", "PD", "C04  PD %"),
        ("C02", "Prodromal", "C02  Prod. %"),
        ("S06", "PSG-RBD", "S06  PSG-RBD %"),
        ("S03", "Genetic", "S03  Genetic %"),
        ("S01", "TD", "S01  Tremor %"),
        ("F05", "Yes", "F05  ADL<80 %"),
    ]
    t_lookup = tasks.set_index("task")
    y_pos = np.arange(len(shift_specs))[::-1]
    for yi, (tid, cls_key, lbl) in zip(y_pos, shift_specs):
        row = t_lookup.loc[tid]
        first_p = json.loads(row.first_dev_target_summary)["shares_pct"][cls_key]
        last_p = json.loads(row.latest_dev_target_summary)["shares_pct"][cls_key]
        eval_p = json.loads(row.eval_target_summary)["shares_pct"][cls_key]
        is_big = abs(last_p - first_p) >= 5.0
        col = rust if is_big else teal
        ax_b.annotate(
            "",
            xy=(last_p, yi),
            xytext=(first_p, yi),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, mutation_scale=8.5, shrinkA=3.5, shrinkB=3.5),
            zorder=2,
        )
        ax_b.scatter([first_p], [yi], facecolors="white", edgecolors=navy, linewidths=1.25, s=32, zorder=4)
        ax_b.scatter([last_p], [yi], color=col, s=34, zorder=4)
        ax_b.scatter([eval_p], [yi], marker="D", color=amber, edgecolor="#5e3c08", linewidths=0.6, s=28, zorder=5)

    ax_b.axhline(1.5, color="#c8d6df", linestyle=":", linewidth=0.9)
    ax_b.text(98.5, 1.32, "Stable controls", fontsize=6.4, color=teal, ha="right", va="top", style="italic")
    ax_b.set_yticks(y_pos, [display_task_text(lbl, task_labels) for _, _, lbl in shift_specs], fontsize=7.4)
    ax_b.set_xlim(0, 102)
    ax_b.set_ylim(-0.6, 7.6)
    ax_b.set_xlabel("Target class share (%)", fontsize=8.3)
    ax_b.set_title("B. Class-proportion shift vs fixed eval", loc="left", fontsize=8.9, weight="bold", color="#22313b")
    ax_b.grid(axis="x", color="#e0eaef", linewidth=0.6)
    ax_b.set_axisbelow(True)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.tick_params(axis="x", labelsize=7.5)
    leg_b = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=navy, markeredgewidth=1.2, markersize=4.8, label="First-cut dev"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=amber, markeredgecolor="#5e3c08", markeredgewidth=0.6, markersize=4.5, label="Fixed eval"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=rust, markersize=4.8, label="Aug 2026 (shifted)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=teal, markersize=4.8, label="Aug 2026 (stable)"),
    ]
    ax_b.legend(handles=leg_b, loc="upper right", frameon=True, facecolor="white", edgecolor="#dce7ec", fontsize=6.2, handletextpad=0.28, borderpad=0.28)

    # --- Panel C: Brier relative change (%) vs AUROC change across all 18 classification tasks ---
    cls = data.loc[data.metric != "mae"].copy()
    cls["dbrier_pct"] = 100.0 * (cls.last_cut - cls.first_cut) / cls.first_cut
    cls["dauroc"] = cls.last_auroc - cls.first_auroc

    y_lo, y_hi = -0.056, 0.066
    y0_frac = (0.0 - y_lo) / (y_hi - y_lo)
    ax_c.axvspan(-16.0, 0.0, ymin=y0_frac, ymax=1.0, color="#eaf6f2", alpha=0.55, zorder=0)
    ax_c.axvspan(0.0, 22.8, ymin=y0_frac, ymax=1.0, color="#fef6e7", alpha=0.65, zorder=0)
    ax_c.axvspan(0.0, 22.8, ymin=0.0, ymax=y0_frac, color="#fdf0ee", alpha=0.55, zorder=0)
    ax_c.axvline(0.0, color="#6d7e8a", linestyle="--", linewidth=0.9, zorder=1)
    ax_c.axhline(0.0, color="#6d7e8a", linestyle="--", linewidth=0.9, zorder=1)

    fam_colors = {
        "Cognition": "#2b6cb0",
        "Motor and function": teal,
        "Enrollment groups": rust,
        "Subtypes": "#7b4ea3",
    }
    fam_markers = {
        "Cognition": "s",
        "Motor and function": "o",
        "Enrollment groups": "D",
        "Subtypes": "^",
    }
    for fam in ["Cognition", "Motor and function", "Enrollment groups", "Subtypes"]:
        sub = cls.loc[cls.family.eq(fam)]
        ax_c.scatter(
            sub.dbrier_pct,
            sub.dauroc,
            color=fam_colors[fam],
            marker=fam_markers[fam],
            s=36,
            edgecolor="white",
            linewidth=0.55,
            label=fam.replace("Motor and function", "Motor/func.").replace("Enrollment groups", "Enrollment"),
            zorder=4,
        )

    # Place labels beside their markers by default. Use dashed elbows for labels
    # that need vertical clearance around nearby points.
    left_side: set[str] = {"S02", "S01", "T1", "C02"}
    vertical_shifts: dict[str, tuple[float, str]] = {
        "C01": (0.0124, "left"),
        "F10": (0.0124, "right"),
        "C05": (-0.0062, "right"),
        "C04": (-0.0088, "right"),
    }
    elbow_dx = 1.15  # horizontal dashed elbow length in x-axis units (%)
    text_halo = [patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.88)]
    for r in cls.itertuples():
        x_m = float(r.dbrier_pct)
        y_m = float(r.dauroc)
        weight = "bold" if r.task in {"S09", "C03", "C04", "C02", "S06", "F05", "F08"} else "normal"
        if r.task in vertical_shifts:
            y_lbl, side = vertical_shifts[r.task]
            sgn = 1.0 if side == "right" else -1.0
            ax_c.plot(
                [x_m, x_m, x_m + sgn * elbow_dx],
                [y_m, y_lbl, y_lbl],
                linestyle=(0, (2.2, 1.4)),
                color=fam_colors[r.family],
                alpha=0.82,
                linewidth=0.85,
                solid_joinstyle="miter",
                zorder=3,
            )
            anchor_x = x_m + sgn * elbow_dx
            text_dx_pt = sgn * 1.2
            ha = "left" if side == "right" else "right"
        elif r.task in left_side:
            y_lbl = y_m
            anchor_x = x_m
            text_dx_pt = -4.8
            ha = "right"
        else:
            y_lbl = y_m
            anchor_x = x_m
            text_dx_pt = 4.8
            ha = "left"
        ax_c.annotate(
            display_task_text(r.task, task_labels),
            xy=(anchor_x, y_lbl),
            xytext=(text_dx_pt, 0.0),
            textcoords="offset points",
            fontsize=6.5,
            weight=weight,
            color="#1e2c36",
            ha=ha,
            va="center_baseline",
            path_effects=text_halo,
            zorder=6,
        )

    ax_c.set_xlim(-15.5, 22.2)
    ax_c.set_ylim(y_lo, y_hi)
    ax_c.text(-14.8, 0.062, "Joint gain\n(Brier \u2193, AUROC \u2191)", fontsize=6.6, color=teal, weight="bold", va="top")
    ax_c.text(21.5, 0.062, "Prior-shift dissociation\n(Brier \u2191, AUROC \u2191)", fontsize=6.6, color="#9e6210", weight="bold", va="top", ha="right")
    ax_c.text(21.5, -0.052, "Brier \u2191 & AUROC \u2193\n(case-mix shift)", fontsize=6.5, color=rust, weight="bold", va="bottom", ha="right")
    ax_c.set_xlabel("Raw release Brier change (%)", fontsize=8.3)
    ax_c.set_ylabel("\u0394AUROC (latest \u2212 first)", fontsize=8.3)
    ax_c.set_title("C. Raw Brier vs \u0394AUROC (18 cls tasks)", loc="left", fontsize=8.9, weight="bold", color="#22313b")
    ax_c.grid(color="#e0eaef", linewidth=0.55)
    ax_c.set_axisbelow(True)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.tick_params(labelsize=7.5)
    ax_c.legend(
        loc="lower left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#dce7ec",
        fontsize=6.1,
        handletextpad=0.25,
        columnspacing=0.6,
        borderpad=0.28,
    )

    # --- Panel D (NEW): Zero-leakage prior/mean-shift counterfactual decomposition ---
    if ax_d is not None:
        decomp_specs = [
            ("S09", "S09  RBD vs hyposmia"),
            ("C03", "C03  PD vs prodromal"),
            ("C04", "C04  3-group enroll."),
            ("C02", "C02  Prodromal vs HC"),
            ("S06", "S06  PSG-RBD vs hypo."),
            ("S04", "S04  3-group genetic"),
            ("N07", "N07  BL UPSIT (reg.)"),
            ("S03", "S03  Sporadic vs gen."),
            ("T1", "T1  2y cog. state"),
            ("F02", "F02  2y MDS-UPDRS IV"),
        ]
        d_map = data.set_index("task")
        y_d = np.arange(len(decomp_specs))[::-1]
        ax_d.axvspan(-3.0, 3.0, color="#edf4f7", alpha=0.75, zorder=0)
        ax_d.axvline(0.0, color="#4d606d", linestyle="--", linewidth=0.95, zorder=1)
        ax_d.axhline(1.5, color="#c8d6df", linestyle=":", linewidth=0.9, zorder=1)

        for yi, (tid, lbl) in zip(y_d, decomp_specs):
            r = d_map.loc[tid]
            raw_v = float(r.first_to_last_pct)
            adj_v = float(r.adj_first_to_last_pct)
            expl = float(r.prior_explained_pct) if pd.notna(r.prior_explained_pct) else np.nan
            a_col = teal if adj_v < 0 else (navy if abs(adj_v) < abs(raw_v) - 1.0 else slate)
            if abs(raw_v - adj_v) >= 0.4:
                ax_d.annotate(
                    "",
                    xy=(adj_v, yi),
                    xytext=(raw_v, yi),
                    arrowprops=dict(arrowstyle="-|>", color=a_col, lw=1.35, mutation_scale=8.5, shrinkA=3.5, shrinkB=3.5),
                    zorder=3,
                )
            ax_d.scatter([raw_v], [yi], color=rust, s=34, zorder=4)
            ax_d.scatter([adj_v], [yi], marker="D", color=a_col, edgecolor="white", linewidths=0.5, s=34, zorder=5)
            if np.isfinite(expl):
                tag = f"{expl:.0f}%" if expl >= 0 else "0%"
                ax_d.text(19.6, yi, tag, fontsize=6.5, color=a_col if expl >= 50 else rust, va="center", ha="right", weight="bold" if expl >= 80 else "normal")

        ax_d.text(19.6, 9.72, "Prior %", fontsize=6.5, color="#33434e", ha="right", va="bottom", weight="bold")
        ax_d.text(16.6, 1.34, "Residual drift (non-prior)", fontsize=6.2, color=rust, ha="right", va="top", style="italic")
        ax_d.set_yticks(y_d, [display_task_text(lbl, task_labels) for _, lbl in decomp_specs], fontsize=7.0)
        ax_d.set_xlim(-8.5, 20.5)
        ax_d.set_ylim(-0.6, 10.15)
        ax_d.set_xlabel("First \u2192 latest error change (%)", fontsize=8.3)
        ax_d.set_title("D. Raw vs prior/mean-adjusted shift", loc="left", fontsize=8.9, weight="bold", color="#22313b")
        ax_d.grid(axis="x", color="#e0eaef", linewidth=0.55)
        ax_d.set_axisbelow(True)
        ax_d.spines[["top", "right"]].set_visible(False)
        ax_d.tick_params(axis="x", labelsize=7.5)
        leg_d = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=rust, markersize=5.0, label="Raw error change"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=navy, markeredgecolor="white", markersize=5.0, label="Prior/mean-adjusted"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=teal, markeredgecolor="white", markersize=5.0, label="Flipped to gain (<0%)"),
        ]
        ax_d.legend(handles=leg_d, loc="lower left", frameon=True, facecolor="white", edgecolor="#dce7ec", fontsize=6.0, handletextpad=0.25, borderpad=0.26)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "cohort_shift_and_divergence.pdf")
    plt.close(fig)


def make_figure3_global_overview(
    data: pd.DataFrame, figures_dir: Path, task_labels: dict[str, str] | None = None
) -> None:
    """Figure 3: Three-column 39-task panorama across release drift, chronological gain, and full-pool AUROC / R^2."""
    ordered_rows = []
    for fam in FAMILY_ORDER:
        sub = data.loc[data.family.eq(fam)]
        for r in sub.itertuples():
            ordered_rows.append(r)
    if len(ordered_rows) != len(data):
        ordered_rows = list(data.itertuples())

    n_tasks = len(ordered_rows)
    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(8.6, 9.25),
        sharey=True,
        gridspec_kw={"width_ratios": [1.16, 1.16, 1.04], "wspace": 0.14},
    )
    fig.subplots_adjust(left=0.095, right=0.795, top=0.915, bottom=0.075)

    navy = "#274a69"
    teal = "#176957"
    rust = "#b24337"
    amber = "#c27d1a"
    reg_col = "#2f7d65"
    cls_col = "#274a69"

    y_vals = np.arange(n_tasks)[::-1]

    # Family bands and dividers
    fam_spans = []
    start = 0
    for idx in range(1, n_tasks + 1):
        if idx == n_tasks or ordered_rows[idx].family != ordered_rows[start].family:
            fam_spans.append((ordered_rows[start].family, start, idx - 1))
            start = idx

    for f_i, (fam, s_idx, e_idx) in enumerate(fam_spans):
        y_top = y_vals[s_idx] + 0.5
        y_bot = y_vals[e_idx] - 0.5
        if f_i % 2 == 1:
            for ax in (ax1, ax2, ax3):
                ax.axhspan(y_bot, y_top, color="#f4f8fa", zorder=0)
        if f_i > 0:
            for ax in (ax1, ax2, ax3):
                ax.axhline(y_top, color="#cfdbe3", linewidth=0.75, zorder=1)

    # Panel A: Official release first->last (%) and full spread (%)
    ax1.axvspan(-3.0, 3.0, color="#edf4f7", alpha=0.7, zorder=0)
    ax1.axvline(0.0, color="#4d606d", linewidth=0.9, zorder=2)
    for xv in (-5.0, 5.0):
        ax1.axvline(xv, color="#b6c6d0", linestyle=":", linewidth=0.75, zorder=1)

    for yi, r in zip(y_vals, ordered_rows):
        col = teal if r.first_to_last_pct < 0 else rust
        # Spread tick bar in muted gray on positive side or subtle underlay
        ax1.hlines(yi, 0.0, r.first_to_last_pct, color=col, linewidth=2.1, zorder=3)
        ax1.scatter([r.first_to_last_pct], [yi], color=col, s=22, zorder=4)
        # Small open marker for full spread magnitude if > |first_to_last| + 1.0
        if r.release_spread > abs(r.first_to_last_pct) + 1.0:
            sign = -1.0 if r.first_to_last_pct < 0 else 1.0
            ax1.scatter([sign * r.release_spread], [yi], facecolors="white", edgecolors=col, linewidths=0.9, s=16, zorder=4)

    ax1.set_xlim(-15.5, 21.0)
    ax1.set_xlabel("First \u2192 last error change (%)", fontsize=8.3)
    ax1.set_title("A. Official release shift", loc="left", fontsize=9.0, weight="bold", color=navy)

    # Panel B: Chronological growth (Small -> Full % and Penultimate -> Full %)
    ax2.axvline(0.0, color="#4d606d", linewidth=0.9, zorder=2)
    for yi, r in zip(y_vals, ordered_rows):
        if pd.isna(r.small_to_full_pct):
            ax2.text(
                -6.5,
                yi,
                "Single tier (SWEDD Jan 2024)",
                fontsize=6.4,
                color=rust,
                va="center",
                ha="center",
                style="italic",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.0),
                zorder=4,
            )
            continue
        col = teal if r.small_to_full_pct < 0 else rust
        ax2.hlines(yi, 0.0, r.small_to_full_pct, color=col, linewidth=2.0, zorder=3)
        ax2.scatter([r.small_to_full_pct], [yi], color=col, s=22, zorder=4)
        if pd.notna(r.last_step_pct):
            dcol = "#124c3e" if r.last_step_pct < 0 else amber
            ax2.scatter([r.last_step_pct], [yi], marker="D", color=dcol, edgecolor="white", linewidths=0.45, s=18, zorder=5)

    ax2.set_xlim(-31.0, 17.5)
    ax2.set_xlabel("Chronological error change (%)", fontsize=8.3)
    ax2.set_title("B. Sample-size growth", loc="left", fontsize=9.0, weight="bold", color=teal)

    # Panel C: Full-pool secondary benchmark (AUROC for classification, R^2 for regression)
    ax3.axvline(0.0, color="#4d606d", linewidth=0.9, zorder=2)
    ax3.axvline(0.5, color="#9ab0be", linestyle="--", linewidth=0.85, zorder=1)
    for yi, r in zip(y_vals, ordered_rows):
        if pd.notna(r.full_auroc):
            val = r.full_auroc
            ax3.barh(yi, val, height=0.60, color=cls_col, alpha=0.86, zorder=3)
            ax3.text(val + 0.018, yi, f"{val:.2f}", fontsize=6.5, color=cls_col, va="center", ha="left", weight="bold")
        elif pd.notna(r.full_r2):
            val = r.full_r2
            col = reg_col if val >= 0 else rust
            ax3.barh(yi, val, height=0.60, color=col, alpha=0.82, zorder=3)
            if val >= 0:
                ax3.text(val + 0.018, yi, f"{val:.2f}", fontsize=6.5, color=col, va="center", ha="left")
            else:
                ax3.text(0.018, yi, f"\u2212{abs(val):.2f}", fontsize=6.5, color=rust, va="center", ha="left")

    ax3.set_xlim(-0.10, 1.03)
    ax3.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax3.set_xlabel("Aug 2026 AUROC or R\u00b2", fontsize=8.3)
    ax3.set_title("C. Full-pool benchmark", loc="left", fontsize=9.0, weight="bold", color="#22313b")

    # Left-column family names aligned to their row groups
    short_fam = {
        "Cognition": "Cognition",
        "Motor and function": "Motor &\nfunction",
        "Enrollment groups": "Enrollment\ngroups",
        "Non-motor symptoms": "Non-motor\nsymptoms",
        "Subtypes": "Subtypes",
        "Biomarkers": "Biomarkers",
        "Prodromal outcomes": "Prodromal",
    }
    family_coordinates = blended_transform_factory(fig.transFigure, ax1.transData)
    for fam, s_idx, e_idx in fam_spans:
        y_mid = 0.5 * (y_vals[s_idx] + y_vals[e_idx])
        ax1.text(
            0.023,
            y_mid,
            short_fam.get(fam, fam),
            transform=family_coordinates,
            fontsize=6.9,
            weight="bold",
            color="#3e515e",
            ha="left",
            va="center",
            linespacing=1.15,
            clip_on=False,
        )

    # Task identifiers and names in the right column
    labels = [display_task_text(SHORT_LABELS.get(r.task, r.task), task_labels) for r in ordered_rows]
    ax3.set_yticks(y_vals, labels, fontsize=7.0)
    ax1.set_ylim(-0.7, n_tasks - 0.3)

    for ax in (ax1, ax2, ax3):
        ax.grid(axis="x", color="#e4edf2", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.tick_params(axis="y", length=0, labelleft=False, labelright=ax is ax3)
    ax3.tick_params(axis="y", pad=5)

    # Compact legend strip at top
    leg_handles = [
        Line2D([0], [0], color=teal, marker="o", markersize=4.5, lw=1.8, label="Error decreased"),
        Line2D([0], [0], color=rust, marker="o", markersize=4.5, lw=1.8, label="Error increased"),
        Line2D([0], [0], color="w", marker="o", markerfacecolor="white", markeredgecolor="#4d606d", markeredgewidth=0.9, markersize=4.5, label="Full spread (A)"),
        Line2D([0], [0], color="w", marker="D", markerfacecolor=amber, markeredgecolor="white", markersize=4.8, label="Final doubling step (B)"),
        Patch(facecolor=cls_col, alpha=0.86, label="AUROC (Cls, C)"),
        Patch(facecolor=reg_col, alpha=0.82, label="R\u00b2 (Reg, C)"),
    ]
    fig.legend(
        handles=leg_handles,
        loc="upper center",
        bbox_to_anchor=(0.445, 0.985),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#dce7ec",
        fontsize=7.3,
        handlelength=1.4,
        columnspacing=1.0,
        borderpad=0.35,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "global_39task_overview.pdf")
    plt.close(fig)


def panel_figure(
    records: dict, keys: list[str], main: bool = False, task_labels: dict[str, str] | None = None
):
    n_rows = len(keys)
    has_adj = any(
        ("prior_adj_error" in v.columns and v.prior_adj_error.notna().any())
        for _, v, _ in records.values()
    )
    if main:
        fig_h = 7.65 if has_adj else 7.45
        hspace = 0.78
        bottom = 0.080
        top = 0.902 if has_adj else 0.930
        title_fs = 8.9
        sub_fs = 8.1
        tick_fs = 7.9
    else:
        fig_h = min(8.85, max(7.20, 1.26 * n_rows + 0.85))
        hspace = 0.64 if n_rows >= 7 else (0.68 if n_rows == 6 else 0.74)
        bottom = 0.062
        top = 0.902 if has_adj else 0.945
        title_fs = 8.1 if n_rows >= 7 else 8.4
        sub_fs = 7.4 if n_rows >= 7 else 7.7
        tick_fs = 7.3 if n_rows >= 7 else 7.6

    fig, axes = plt.subplots(n_rows, 2, figsize=(8.1, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.975, bottom=bottom, top=top, wspace=0.28, hspace=hspace)
    fig.text(0.30, 0.980, "A. Official curated releases", ha="center", color="#274a69", weight="bold", fontsize=9.2)
    fig.text(0.755, 0.980, "B. Chronological training sizes", ha="center", color="#176957", weight="bold", fontsize=9.2)

    if has_adj:
        leg_handles = [
            Line2D([0], [0], color="#274a69", marker="o", markersize=3.2, lw=1.35, label="Raw release error"),
            Line2D([0], [0], color="#176957", marker="o", markersize=3.2, lw=1.35, label="Raw chronological mean"),
            Line2D([0], [0], color="#c27d1a", linestyle="--", marker="s", markersize=2.6, lw=1.15, label="Prior/mean-adjusted"),
            Patch(facecolor="#176957", alpha=0.20, label="95% tie-order band"),
        ]
        fig.legend(
            handles=leg_handles,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.962),
            ncol=4,
            frameon=True,
            facecolor="white",
            edgecolor="#dce7ec",
            fontsize=6.9,
            handlelength=1.5,
            columnspacing=0.9,
            borderpad=0.28,
        )

    has_auroc = any("auroc" in v.columns for _, v, _ in records.values())
    has_r2 = any("r2" in v.columns for _, v, _ in records.values())

    for key, (left, right) in zip(keys, axes):
        row, v, n = records[key]
        dates = pd.to_datetime(v.version, format="%Y%m%d")
        steps = np.arange(len(n))
        left.plot(dates, v.observed_release_error, "-o", color="#274a69", markersize=2.9, linewidth=1.3, zorder=3)
        if has_adj and "prior_adj_error" in v.columns and v.prior_adj_error.notna().any():
            left.plot(
                dates,
                v.prior_adj_error,
                "--s",
                color="#c27d1a",
                markersize=2.1,
                linewidth=1.15,
                alpha=0.92,
                zorder=4,
            )

        valid_mask = (n.valid_rounds > 0).to_numpy()
        if valid_mask.any():
            right.fill_between(
                steps[valid_mask],
                n.error_p025.to_numpy()[valid_mask],
                n.error_p975.to_numpy()[valid_mask],
                color="#176957",
                alpha=0.20,
                linewidth=0,
                zorder=1,
            )
        right.plot(steps, n.observed_error, "-o", color="#176957", markersize=2.9, linewidth=1.3, zorder=3)
        if has_adj and "prior_adj_error_mean" in n.columns and n.prior_adj_error_mean.notna().any():
            right.plot(
                steps,
                n.prior_adj_error_mean,
                "--s",
                color="#c27d1a",
                markersize=2.1,
                linewidth=1.15,
                alpha=0.92,
                zorder=4,
            )

        partial_mask = n.status.eq("partial_missing_class").to_numpy()
        if partial_mask.any():
            right.scatter(
                steps[partial_mask],
                n.observed_error.to_numpy()[partial_mask],
                facecolors="white",
                edgecolors="#a64f46",
                linewidths=1.05,
                s=23,
                zorder=5,
            )

        ticks = date_ticks(dates)
        left.set_xticks(ticks, ticks.dt.strftime("%Y-%m"))
        right.set_xticks(steps, [f"{count:,}" for count in n.training_n])
        series_for_ylim = [v.observed_release_error, n.observed_error, n.error_p025, n.error_p975]
        if has_adj and "prior_adj_error" in v.columns:
            series_for_ylim.append(v.prior_adj_error)
        if has_adj and "prior_adj_error_mean" in n.columns:
            series_for_ylim.append(n.prior_adj_error_mean)
        values = pd.concat(series_for_ylim).dropna()
        vmin, vmax = float(values.min()), float(values.max())
        pad = max(0.14 * (vmax - vmin), abs(vmin) * 0.002, 1e-6)
        wrap_w = 56 if main else 54
        task_label = display_task_text(key, task_labels)
        left.set_title(textwrap.fill(f"{task_label}  {row.title}", wrap_w), loc="left", fontsize=title_fs, weight="bold", pad=3.5)

        bench_str = ""
        if row.primary_metric != "mae" and has_auroc and pd.notna(v.auroc.iloc[-1]):
            ap_txt = f", AUPRC {float(v.auprc.iloc[-1]):.3f}" if ("auprc" in v.columns and pd.notna(v.auprc.iloc[-1])) else ""
            bench_str = f"  |  AUROC {v.auroc.iloc[0]:.3f}\u2192{v.auroc.iloc[-1]:.3f}{ap_txt}"
        elif row.primary_metric == "mae" and has_r2 and pd.notna(v.r2.iloc[-1]):
            r2_last = float(v.r2.iloc[-1])
            r2_txt = f"\u2212{abs(r2_last):.2f}" if r2_last < 0 else f"{r2_last:.2f}"
            nmae_txt = f", NMAE {float(v.nmae.iloc[-1]):.2f}" if ("nmae" in v.columns and pd.notna(v.nmae.iloc[-1])) else ""
            rmse_txt = f", RMSE {float(v.rmse.iloc[-1]):.2f}" if (not nmae_txt and "rmse" in v.columns and pd.notna(v.rmse.iloc[-1])) else ""
            bench_str = f"  |  R\u00b2={r2_txt}{nmae_txt}{rmse_txt}"
        right.set_title(f"Eval n={row.evaluation_n}{bench_str}", loc="left", fontsize=sub_fs, color="#33434e", pad=3.5)
        left.set_ylabel(
            {"brier": "Brier score", "multiclass_brier": "Multiclass Brier", "mae": "MAE"}[row.primary_metric],
            fontsize=8.2 if main else 7.7,
        )
        for axis in (left, right):
            axis.set_ylim(vmin - pad, vmax + pad)
            axis.grid(axis="y", color="#dce7ec", linewidth=0.55)
            axis.set_axisbelow(True)
            axis.tick_params(length=2.8, labelsize=tick_fs)
            axis.spines[["top", "right"]].set_visible(False)

        missing = n.status.eq("missing_training_class").to_numpy()
        if missing.any():
            right.scatter(
                steps[missing],
                np.full(missing.sum(), vmin - pad * 0.52),
                marker="x",
                color="#a64f46",
                s=20,
                zorder=5,
            )
    axes[-1, 0].set_xlabel("Curated extract date", fontsize=8.5)
    axes[-1, 1].set_xlabel("Training participants", fontsize=8.5)
    return fig


def make_figures(
    tasks: pd.DataFrame,
    versions: pd.DataFrame,
    size: pd.DataFrame,
    data: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    main_keys: list[str],
    task_labels: dict[str, str] | None = None,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42})
    records = {row.task: (row, v, n) for row, v, n in pair_trajectories(tasks, versions, size)}
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 2 (cohort expansion, class-proportion shift, Brier vs AUROC, prior-shift decomposition)
    if len(tasks) >= 18 and "auroc" in versions.columns:
        make_figure2_cohort_and_shift(tasks, versions, data, results_dir, figures_dir, task_labels)

    # Figure 3 (39-task panorama)
    if len(tasks) >= 10:
        make_figure3_global_overview(order_for_display(data, task_labels), figures_dir, task_labels)

    # Figure 4 (four representative tasks)
    fig = panel_figure(records, main_keys, main=True, task_labels=task_labels)
    fig.savefig(figures_dir / "chronological_examples.pdf")
    plt.close(fig)

    # Compact family-grouped supplementary figures (S1-S7)
    all_keys_present = set(records.keys())
    grouped_keys = {k for _, _, ks in SUPP_FAMILY_GROUPS for k in ks}
    if grouped_keys.issubset(all_keys_present):
        for fname, _, ks in SUPP_FAMILY_GROUPS:
            fig = panel_figure(records, ks, main=False, task_labels=task_labels)
            fig.savefig(results_dir / fname)
            plt.close(fig)
    else:
        # Fallback for small synthetic test datasets
        order = list(records)
        fig = panel_figure(records, order, main=False, task_labels=task_labels)
        fig.savefig(results_dir / "supp_fig_s1.pdf")
        plt.close(fig)


def write_summary_json(
    tasks: pd.DataFrame,
    versions: pd.DataFrame,
    size: pd.DataFrame,
    data: pd.DataFrame,
    results_dir: Path,
    paired: pd.DataFrame | None = None,
) -> None:
    multi = data.loc[data.estimable_tiers >= 2]
    cls = data.loc[data.metric != "mae"]
    reg = data.loc[data.metric == "mae"]
    cls_multi = cls.loc[cls.estimable_tiers >= 2]

    m1_m4_summary: dict = {}
    if "adj_release_spread" in data.columns and data.adj_release_spread.notna().any():
        bin_cls = cls.loc[cls.metric == "brier"]
        m1_m4_summary = {
            "raw_release_spread_median_pct": round(float(data.release_spread.median()), 2),
            "prior_adj_release_spread_median_pct": round(float(data.adj_release_spread.median()), 2),
            "prior_adj_release_lower_at_latest": int((data.last_adj_cut < data.first_adj_cut).sum()),
            "prior_adj_release_patterns": data.adj_release_pattern.value_counts().to_dict(),
            "release_auprc_improved": int((cls.last_auprc > cls.first_auprc).sum()) if cls.last_auprc.notna().any() else 0,
            "chrono_auprc_improved_multitier": int((cls_multi.last_auprc > cls_multi.small_auprc).sum()) if cls_multi.small_auprc.notna().any() else 0,
            "release_logloss_improved": int((cls.last_logloss < cls.first_logloss).sum()) if cls.last_logloss.notna().any() else 0,
            "release_adj_logloss_improved": int((cls.last_adj_logloss < cls.first_adj_logloss).sum()) if cls.last_adj_logloss.notna().any() else 0,
            "chrono_logloss_improved_multitier": int((cls_multi.last_logloss < cls_multi.small_logloss).sum()) if cls_multi.small_logloss.notna().any() else 0,
            "binary_full_auprc_range": [round(float(bin_cls.last_auprc.min()), 4), round(float(bin_cls.last_auprc.max()), 4)] if len(bin_cls) else [],
            "binary_full_auprc_lift_range": [round(float(bin_cls.last_auprc_lift.min()), 4), round(float(bin_cls.last_auprc_lift.max()), 4)] if len(bin_cls) else [],
            "full_nmae_range": [round(float(reg.last_nmae.min()), 4), round(float(reg.last_nmae.max()), 4)] if len(reg) and reg.last_nmae.notna().any() else [],
        }
    if paired is not None and len(paired):
        m1_m4_summary["paired_bootstrap_95ci"] = {
            "release_raw_improved_excl0": int((paired.rel_err_ci_hi < 0).sum()),
            "release_raw_worsened_excl0": int((paired.rel_err_ci_lo > 0).sum()),
            "release_adj_improved_excl0": int((paired.rel_adj_err_ci_hi < 0).sum()),
            "release_adj_worsened_excl0": int((paired.rel_adj_err_ci_lo > 0).sum()),
            "latest_prior_recal_gain_excl0": int((paired.rel_last_raw_minus_adj_ci_lo > 0).sum()),
            "final_step_raw_improved_excl0": int((paired.step_err_ci_hi < 0).sum()),
            "final_step_raw_worsened_excl0": int((paired.step_err_ci_lo > 0).sum()),
            "final_step_adj_improved_excl0": int((paired.step_adj_err_ci_hi < 0).sum()),
            "final_step_adj_worsened_excl0": int((paired.step_adj_err_ci_lo > 0).sum()),
            "earliest_to_full_raw_improved_excl0": int((paired.span_err_ci_hi < 0).sum()),
            "earliest_to_full_raw_worsened_excl0": int((paired.span_err_ci_lo > 0).sum()),
        }

    summary = {
        "tasks": int(len(tasks)),
        "official_release_points": int(len(versions)),
        "chronological_stages": int(len(size)),
        "full_fit_stages": int(size.status.eq("fit").sum()),
        "partial_missing_class_stages": int(size.status.eq("partial_missing_class").sum()),
        "missing_training_class_stages": int(size.status.eq("missing_training_class").sum()),
        "stages_at_20230612": int(size.verified_cut.eq("20230612").sum()),
        "tasks_starting_at_20230612": int(size.groupby("task").verified_cut.first().eq("20230612").sum()),
        "split_stages": int(size.boundary_group_split.sum()),
        "release_lower_at_latest": int((data.last_cut < data.first_cut).sum()),
        "release_higher_at_latest": int((data.last_cut > data.first_cut).sum()),
        "release_patterns": data.release_pattern.value_counts().to_dict(),
        "multi_tier_tasks": int(len(multi)),
        "full_lower_than_small_mean": int((multi.large_error < multi.small_error).sum()),
        "full_higher_than_small_mean_tasks": multi.loc[multi.large_error > multi.small_error, "task"].tolist(),
        "directions": data.direction.value_counts().to_dict(),
        "declining_tasks": data.loc[data.direction.eq("Declining"), "task"].tolist(),
        "early_narrow_count": int(data.narrow_from.notna().sum()),
        "early_narrow_tasks": {
            row.task: int(row.narrow_from) for row in data.loc[data.narrow_from.notna()].itertuples()
        },
        "eval_n_range": [int(tasks.evaluation_n.min()), int(tasks.evaluation_n.max())],
        "secondary_summary": {
            "classification_tasks": int(len(cls)),
            "regression_tasks": int(len(reg)),
            "release_auroc_improved": int((cls.last_auroc > cls.first_auroc).sum()) if cls.full_auroc.notna().any() else 0,
            "chrono_auroc_improved_multitier": int((cls_multi.full_auroc > cls_multi.small_auroc).sum()) if cls_multi.full_auroc.notna().any() else 0,
            "full_auroc_range": [round(float(cls.full_auroc.min()), 4), round(float(cls.full_auroc.max()), 4)] if cls.full_auroc.notna().any() else [],
            "release_r2_improved": int((reg.last_r2 > reg.first_r2).sum()) if reg.full_r2.notna().any() else 0,
            "release_rmse_improved": int((reg.last_rmse < reg.first_rmse).sum()) if reg.full_rmse.notna().any() else 0,
            "chrono_r2_improved": int((reg.full_r2 > reg.small_r2).sum()) if reg.full_r2.notna().any() else 0,
            "chrono_rmse_improved": int((reg.full_rmse < reg.small_rmse).sum()) if reg.full_rmse.notna().any() else 0,
            "full_r2_range": [round(float(reg.full_r2.min()), 4), round(float(reg.full_r2.max()), 4)] if reg.full_r2.notna().any() else [],
        },
        "m1_m4_summary": m1_m4_summary,
    }
    (results_dir / "trajectory_summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--main-tasks", nargs="+", default=["T2", "F05", "N01", "S09"])
    parser.add_argument(
        "--task-display-map",
        type=Path,
        default=LOCAL_DISPLAY_MAP if LOCAL_DISPLAY_MAP.is_file() else None,
        help="JSON object mapping original task IDs to display labels, in presentation order",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    tasks, versions, size = load_inputs(results_dir)
    paired = load_paired(results_dir)
    data = describe_trajectories(tasks, versions, size)
    task_labels = None
    if args.task_display_map is not None:
        task_labels = json.loads(args.task_display_map.read_text(encoding="utf-8"))
        if (
            not isinstance(task_labels, dict)
            or set(task_labels) != set(tasks.task)
            or any(not isinstance(label, str) or not re.fullmatch(r"[A-Z]+[0-9]{2}", label) for label in task_labels.values())
            or len(set(task_labels.values())) != len(task_labels)
            or not task_labels.keys().isdisjoint(task_labels.values())
        ):
            parser.error("task display map must assign a unique new label to every task")
    make_tables(order_for_display(data, task_labels), size, results_dir, paired=paired, task_labels=task_labels)
    make_task_catalog(order_for_display(tasks, task_labels), results_dir, task_labels)
    make_figures(tasks, versions, size, data, results_dir, figures_dir, args.main_tasks, task_labels)
    write_summary_json(tasks, versions, size, data, results_dir, paired=paired)


if __name__ == "__main__":
    main()
