"""Generate publication tables and trajectory figures from unified aggregate outputs."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

CATALOG_NAMES = {
    "T1": "Cognitive state (normal vs MCI or dementia); PD with normal cognition at baseline",
    "T2": "MoCA",
    "T3": "MDS-UPDRS II",
    "T4": "MDS-UPDRS III (OFF or untreated)",
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
    "P02": "MDS-UPDRS III (OFF or untreated) in prodromal participants",
}

CORE_FEATURES = ["age", "SEX", "EDUCYRS", "gds", "ess", "scopa"]
PREDICTOR_OVERRIDES = {
    "T1": "Age, MoCA",
    "T2": "Age, MoCA",
    "T3": "Age, MDS-UPDRS II",
    "T4": "Age, MDS-UPDRS III (OFF or untreated)",
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


def describe_trajectories(tasks: pd.DataFrame, versions: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row, v, n in pair_trajectories(tasks, versions, size):
        errors = v.observed_release_error.to_numpy()
        fitted = n.loc[n.valid_rounds > 0]
        trajectory = fitted.observed_error.to_numpy()
        sizes = fitted.training_n.to_numpy()
        has_multi = len(trajectory) >= 2
        consecutive = np.diff(trajectory) if has_multi else np.array([])
        relative = 100.0 * np.ptp(errors) / errors[0]
        plateau = next(
            (int(sizes[i]) for i in range(len(sizes) - 2) if 100.0 * np.ptp(trajectory[i:]) / trajectory[i] <= 2.0),
            None,
        )
        records.append(
            {
                "task": row.task,
                "family": row.family,
                "metric": row.primary_metric,
                "first_cut": float(errors[0]),
                "last_cut": float(errors[-1]),
                "release_spread": float(relative),
                "release_pattern": "Narrow" if relative <= 3.0 else ("Wide" if relative >= 5.0 else "Intermediate"),
                "estimable_tiers": len(fitted),
                "small_n": int(sizes[0]) if has_multi else None,
                "small_error": float(trajectory[0]) if has_multi else np.nan,
                "small_p025": float(fitted.error_p025.iloc[0]) if has_multi else np.nan,
                "small_p975": float(fitted.error_p975.iloc[0]) if has_multi else np.nan,
                "large_n": int(sizes[-1]),
                "large_error": float(trajectory[-1]),
                "direction": (
                    "Declining"
                    if (has_multi and (consecutive < 0).all())
                    else ("Single tier" if not has_multi else "Reversals")
                ),
                "narrow_from": plateau,
            }
        )
    data = pd.DataFrame(records)
    assert len(data) == len(tasks)
    assert (data.release_pattern.value_counts().sum()) == len(tasks)
    assert (data.direction.value_counts().sum()) == len(tasks)
    return data


def make_tables(data: pd.DataFrame, size: pd.DataFrame, results_dir: Path) -> None:
    header = (
        r"\begin{tabularx}{\linewidth}{@{}Xrrrrr@{}}" "\n"
        r"\toprule" "\n"
        r"\textbf{Task family} & \textbf{Tasks} & \textbf{Narrow} & \textbf{Wide} &"
        r" \textbf{Declining} & \textbf{Early narrow}" "\\\\\n"
        r"\midrule"
    )
    rows = []
    for family, group in data.groupby("family", sort=False):
        rows.append(
            f"{family} & {len(group)} & {sum(group.release_pattern.eq('Narrow'))} & "
            f"{sum(group.release_pattern.eq('Wide'))} & {sum(group.direction.eq('Declining'))} & "
            f"{sum(group.narrow_from.notna())} " + r"\\"
        )
    rows.append(
        f"\\midrule Total & {len(data)} & {sum(data.release_pattern.eq('Narrow'))} & "
        f"{sum(data.release_pattern.eq('Wide'))} & {sum(data.direction.eq('Declining'))} & "
        f"{sum(data.narrow_from.notna())} " + r"\\"
    )
    (results_dir / "paper_family_table.tex").write_text(
        "\n".join([header, *rows, r"\bottomrule", r"\end{tabularx}", ""]), encoding="utf-8"
    )

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

    spec = r"\begin{longtable}{@{}lrrrrrrr@{}}"
    caption = (
        r"\caption{\textbf{Primary error by task on the two trajectories.} First and last refer to the "
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
        r"\textbf{Task} & \textbf{First cut} & \textbf{Last cut} & \textbf{Spread (\%)} & "
        r"\textbf{Small mean} & \textbf{Small 95\% range} & \textbf{Full} & \textbf{Narrow from} \\" "\n"
        r"\midrule"
    )
    rows = []
    for item in data.itertuples():
        digits = 4 if item.metric == "mae" else 5
        rdigits = 3 if item.metric == "mae" else 4
        narrow = f"{int(item.narrow_from):,}" if pd.notna(item.narrow_from) else "---"
        if pd.notna(item.small_error):
            small_str = f"{item.small_error:.{digits}f}"
            range_str = f"{item.small_p025:.{rdigits}f}--{item.small_p975:.{rdigits}f}"
        else:
            small_str = "---"
            range_str = "---"
        rows.append(
            f"{item.task} & {item.first_cut:.{digits}f} & {item.last_cut:.{digits}f} & "
            f"{item.release_spread:.2f} & {small_str} & {range_str} & "
            f"{item.large_error:.{digits}f} & {narrow} " + r"\\"
        )
    text = [spec, caption, heading, r"\endfirsthead", heading, r"\endhead", r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}", ""]
    (results_dir / "paper_task_results.tex").write_text("\n".join(text), encoding="utf-8")


def make_task_catalog(tasks: pd.DataFrame, results_dir: Path) -> None:
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
                spec = "Core, MDS-UPDRS II and III (OFF or untreated)"
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
        r" In classification rows, counts in parentheses are in the stated class order. For a"
        r" binary milestone or threshold outcome, they list absence then presence."
        r" Core denotes age, sex, years of education, GDS, ESS and SCOPA-AUT."
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
    spec = r"\begin{longtable}{@{}lp{.43\linewidth}lp{.218\linewidth}l@{}}"
    text = [spec, caption, heading, r"\endfirsthead", heading, r"\endhead", r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}", ""]
    (results_dir / "paper_task_catalog.tex").write_text("\n".join(text), encoding="utf-8")


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


def panel_figure(records: dict, keys: list[str], main: bool = False, page: int = 0):
    fig, axes = plt.subplots(len(keys), 2, figsize=(8, 7.25 if main else 8.3), squeeze=False)
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.13 if main else 0.15, top=0.925, wspace=0.35, hspace=0.85)
    fig.text(0.30, 0.975, "A. Official curated releases", ha="center", color="#274a69", weight="bold")
    fig.text(0.75, 0.975, "B. Chronological training sizes", ha="center", color="#176957", weight="bold")
    for key, (left, right) in zip(keys, axes):
        row, v, n = records[key]
        dates = pd.to_datetime(v.version, format="%Y%m%d")
        steps = np.arange(len(n))
        left.plot(dates, v.observed_release_error, "-o", color="#274a69", markersize=3, linewidth=1.3)

        valid_mask = (n.valid_rounds > 0).to_numpy()
        if valid_mask.any():
            right.fill_between(
                steps[valid_mask],
                n.error_p025.to_numpy()[valid_mask],
                n.error_p975.to_numpy()[valid_mask],
                color="#176957",
                alpha=0.20,
                linewidth=0,
            )
        right.plot(steps, n.observed_error, "-o", color="#176957", markersize=3, linewidth=1.3)

        partial_mask = n.status.eq("partial_missing_class").to_numpy()
        if partial_mask.any():
            right.scatter(
                steps[partial_mask],
                n.observed_error.to_numpy()[partial_mask],
                facecolors="white",
                edgecolors="#a64f46",
                linewidths=1.1,
                s=26,
                zorder=4,
            )

        ticks = date_ticks(dates)
        left.set_xticks(ticks, ticks.dt.strftime("%Y-%m"))
        right.set_xticks(steps, [f"{count:,}" for count in n.training_n])
        values = pd.concat([v.observed_release_error, n.observed_error, n.error_p025, n.error_p975]).dropna()
        vmin, vmax = float(values.min()), float(values.max())
        pad = max(0.13 * (vmax - vmin), abs(vmin) * 0.002, 1e-6)
        left.set_title(textwrap.fill(f"{key}  {row.title}", 53), loc="left", fontsize=9, weight="bold")
        right.set_title(f"{row.evaluation_n} evaluation participants", loc="left", fontsize=9)
        left.set_ylabel({"brier": "Brier score", "multiclass_brier": "Multiclass Brier", "mae": "MAE"}[row.primary_metric])
        for axis in (left, right):
            axis.set_ylim(vmin - pad, vmax + pad)
            axis.grid(axis="y", color="#dce7ec", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.tick_params(length=3, labelsize=8)
            axis.spines[["top", "right"]].set_visible(False)

        missing = n.status.eq("missing_training_class").to_numpy()
        if missing.any():
            right.scatter(
                steps[missing],
                np.full(missing.sum(), vmin - pad * 0.52),
                marker="x",
                color="#a64f46",
                s=20,
                zorder=4,
            )
    axes[-1, 0].set_xlabel("Curated extract date")
    axes[-1, 1].set_xlabel("Training participants")
    if not main:
        fig.text(0.115, 0.042, f"Figure S{page}.", fontsize=10, weight="bold")
        fig.text(
            0.218,
            0.042,
            "Shaded band: 2.5th-97.5th percentile across 1,000 tie orders. Open circle: partial; cross: absent class.",
            fontsize=8.2,
            color="#495a65",
        )
    return fig


def make_figures(tasks: pd.DataFrame, versions: pd.DataFrame, size: pd.DataFrame, results_dir: Path, figures_dir: Path, main_keys: list[str]) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42})
    records = {row.task: (row, v, n) for row, v, n in pair_trajectories(tasks, versions, size)}
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig = panel_figure(records, main_keys, main=True)
    fig.savefig(figures_dir / "chronological_examples.pdf")
    plt.close(fig)

    order = list(records)
    with PdfPages(results_dir / "paper_all_trajectories.pdf") as pdf:
        for page, start in enumerate(range(0, len(order), 3), 1):
            fig = panel_figure(records, order[start : start + 3], page=page)
            pdf.savefig(fig)
            plt.close(fig)


def write_summary_json(tasks: pd.DataFrame, versions: pd.DataFrame, size: pd.DataFrame, data: pd.DataFrame, results_dir: Path) -> None:
    multi = data.loc[data.estimable_tiers >= 2]
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
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    tasks, versions, size = load_inputs(results_dir)
    data = describe_trajectories(tasks, versions, size)
    make_tables(data, size, results_dir)
    make_task_catalog(tasks, results_dir)
    make_figures(tasks, versions, size, results_dir, figures_dir, args.main_tasks)
    write_summary_json(tasks, versions, size, data, results_dir)


if __name__ == "__main__":
    main()
