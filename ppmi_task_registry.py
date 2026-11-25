"""Unified PPMI curated-extract loader, task registry, and stratified evaluation partition."""
from __future__ import annotations

import csv
import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile

import openpyxl
from openpyxl.utils import get_column_letter

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parent if _HERE.name == "code" else _HERE
VERSIONS = (
    "20230612",
    "20240129",
    "20240729",
    "20241211",
    "20250321",
    "20250714",
    "20251112",
    "20260223",
    "20260319",
    "20260511",
    "20260810",
)
REFERENCE = VERSIONS[-1]
MIN_RELEASES = 3
RESERVE_SEED = "P12-expanded-20261121"
RESERVE_FRACTION = 0.30

CORE = ("age", "SEX", "EDUCYRS", "gds", "ess", "scopa")
COHORT = {1: "PD", 2: "HC", 3: "SWEDD", 4: "Prodromal"}
COGNITION = {1: "NC", 2: "MCI", 3: "Dementia"}
NO_YES = {0: "No", 1: "Yes"}
MOTOR = {1: "TD", 2: "non-TD"}
GENES = re.compile(r"\b(?:LRRK2|GBA1?|SNCA|PRKN|PINK1|PARK7|VPS35|OTHER GENETIC VARIANT)\b")

FAMILY_LABELS = {
    "cognition": "Cognition",
    "function": "Motor and function",
    "enrollment": "Enrollment groups",
    "nonmotor": "Non-motor symptoms",
    "subtype": "Subtypes",
    "biomarker": "Biomarkers",
    "prodromal": "Prodromal outcomes",
}

BOUNDS = {
    "age": (18, 120),
    "age_at_visit": (18, 125),
    "SEX": (0, 1),
    "EDUCYRS": (0, 40),
    "moca": (0, 30),
    "HVLTRDLY": (0, 12),
    "NHY": (0, 5),
    "MSEADLG": (0, 100),
    "updrs1_score": (0, 52),
    "updrs2_score": (0, 52),
    "updrs3_score": (0, 132),
    "updrs4_score": (0, 24),
    "scopa": (0, 69),
    "gds": (0, 15),
    "ess": (0, 24),
    "stai_trait": (20, 80),
    "rem": (0, 13),
    "upsit": (0, 40),
    "PDAQ27": (0, 108),
    "cogstate": (1, 3),
}


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    title: str
    target: str
    mode: str = "BL"
    cohorts: tuple[int, ...] = (1,)
    labels: tuple[str, ...] = ()
    features: tuple[str, ...] = CORE
    risk: tuple[tuple, ...] = ()
    must_measure: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    qualification: str = ""

    @property
    def columns(self) -> set[str]:
        return {self.target, *self.features, *self.extra, *(f for f, _, _ in self.risk)} - {""}


def classified(id: str, family: str, title: str, target: str, labels: dict, **options) -> Task:
    return Task(id, family, title, target, labels=tuple(labels.values()), **options)


def continuous(id: str, title: str, target: str, family: str, cohorts: tuple[int, ...] = (1,)) -> Task:
    return Task(id, family, title, target, "M24", cohorts, features=("age", target))


def risk_task(id: str, title: str, target: str, relation: str, threshold, family: str = "function") -> Task:
    return Task(
        id,
        family,
        title,
        target,
        "M24",
        labels=("No", "Yes"),
        features=CORE + ("updrs2_score",),
        risk=((target, relation, threshold),),
    )


def cohort_task(id: str, title: str, groups: tuple[int, ...]) -> Task:
    return classified(id, "enrollment", title, "COHORT", {g: COHORT[g] for g in groups}, cohorts=groups)


def conditional(id: str, family: str, title: str, target: str, reason: str, **kwargs) -> Task:
    return Task(id, family, title, target, qualification=reason, **kwargs)


ALL_CANDIDATE_TASKS: tuple[Task, ...] = (
    # Core two-year benchmark tasks integrated under the same unified rules
    Task(
        "T1",
        "cognition",
        "Two-year cognitive state",
        "cogstate",
        mode="M24",
        cohorts=(1,),
        labels=("NC", "MCI or dementia"),
        features=("age", "moca"),
        risk=(("cogstate", "eq", 1),),
    ),
    continuous("T2", "Two-year MoCA", "moca", "cognition"),
    continuous("T3", "Two-year MDS-UPDRS II", "updrs2_score", "function"),
    continuous("T4", "Two-year MDS-UPDRS III (OFF or untreated)", "updrs3_score", "function"),
    # Enrollment groups
    cohort_task("C01", "Baseline PD vs HC enrollment", (1, 2)),
    cohort_task("C02", "Baseline prodromal vs HC enrollment", (4, 2)),
    cohort_task("C03", "Baseline PD vs prodromal enrollment", (1, 4)),
    cohort_task("C04", "Baseline PD vs prodromal vs HC enrollment", (1, 4, 2)),
    cohort_task("C05", "Baseline PD vs SWEDD enrollment", (1, 3)),
    # Cognition
    classified("K01", "cognition", "Baseline normal cognition vs MCI in PD", "cogstate", {1: "NC", 2: "MCI"}),
    classified(
        "K03",
        "cognition",
        "PD baseline NC / MCI / dementia",
        "cogstate",
        COGNITION,
        qualification="Feasibility only: dementia is too sparse for multi-release classification.",
    ),
    classified(
        "K04",
        "cognition",
        "PD NC / MCI / dementia at the month-24 visit",
        "cogstate",
        COGNITION,
        mode="VISIT24",
        features=("age_at_visit", "SEX", "EDUCYRS", "gds", "ess", "scopa"),
        qualification="Feasibility only: sparse dementia counts at month 24.",
    ),
    classified(
        "K05",
        "cognition",
        "PD month-24 dementia vs NC/MCI given baseline MCI",
        "cogstate",
        {1: "NC or MCI", 2: "Dementia"},
        mode="M24",
        risk=(("cogstate", "eq", 2),),
        features=("age", "moca"),
        qualification="Feasibility only: sparse dementia conversion among baseline MCI.",
    ),
    continuous("K06", "Two-year SDMT", "SDMTOTAL", "cognition"),
    continuous("K07", "Two-year animal fluency", "VLTANIM", "cognition"),
    continuous("K08", "Two-year HVLT delayed recall", "HVLTRDLY", "cognition"),
    continuous("K09", "Two-year TMT-B", "TMT_B", "cognition"),
    # Motor and function
    continuous("F01", "Two-year MDS-UPDRS I", "updrs1_score", "function"),
    continuous("F02", "Two-year MDS-UPDRS IV", "updrs4_score", "function"),
    continuous("F03", "Two-year Schwab\u2013England ADL", "MSEADLG", "function"),
    risk_task("F04", "Two-year H&Y stage \u22653 (baseline <3)", "NHY", "lt", 3),
    risk_task("F05", "Two-year Schwab\u2013England ADL <80 (baseline \u226580)", "MSEADLG", "ge", 80),
    risk_task("F06", "Two-year any progression milestone", "pm_any", "eq", 0),
    risk_task("F07", "Two-year functional dependence milestone", "pm_fd_any", "eq", 0),
    risk_task("F08", "Two-year walking or balance milestone", "pm_wb_any", "eq", 0),
    risk_task("F09", "Two-year motor complication milestone", "pm_mc_any", "eq", 0),
    risk_task("F10", "Two-year PD medication or DBS initiation", "PDTRTMNT", "eq", 0),
    continuous("F11", "Two-year PDAQ-27", "PDAQ27", "function"),
    Task(
        "F12",
        "function",
        "PD all-cause death within 24 months",
        "Death_Date",
        "DEATH",
        labels=("Alive through month 24", "Death by month 24"),
        extra=("Death_Status",),
        qualification="Date availability limits the series; requires separate vital-status validation.",
    ),
    # Non-motor symptoms
    continuous("N01", "Two-year SCOPA-AUT", "scopa", "nonmotor"),
    continuous("N02", "Two-year GDS", "gds", "nonmotor"),
    continuous("N03", "Two-year ESS", "ess", "nonmotor"),
    continuous("N04", "Two-year STAI trait anxiety", "stai_trait", "nonmotor"),
    continuous("N05", "Two-year RBDSQ", "rem", "nonmotor"),
    continuous("N06", "Two-year UPSIT", "upsit", "nonmotor"),
    Task("N07", "nonmotor", "Baseline UPSIT olfactory score in PD", "upsit"),
    # Subtypes
    classified("S01", "subtype", "Baseline tremor-dominant vs other subtype", "td_pigd", MOTOR),
    classified(
        "S02",
        "subtype",
        "Two-year tremor-dominant vs other subtype",
        "td_pigd",
        MOTOR,
        mode="M24",
        features=CORE + ("updrs2_score", "updrs3_score"),
    ),
    Task("S03", "subtype", "Baseline sporadic-type vs genetic enrollment", "subgroup", labels=("Sporadic-type", "Genetic")),
    Task(
        "S04",
        "subtype",
        "Baseline sporadic-type vs LRRK2 vs GBA enrollment",
        "subgroup",
        labels=("Sporadic-type", "LRRK2", "GBA"),
    ),
    Task(
        "S05",
        "subtype",
        "Prodromal RBD-only / hyposmia-only / both enrollment subgroup",
        "subgroup",
        cohorts=(4,),
        labels=("RBD", "Hyposmia", "Both"),
    ),
    Task(
        "S06",
        "subtype",
        "Baseline PSG-confirmed RBD vs hyposmia alone",
        "analytic_subgroup",
        cohorts=(4,),
        labels=("PSG-RBD", "Hyposmia alone"),
    ),
    Task(
        "S07",
        "subtype",
        "Mild motor-predominant / intermediate / diffuse malignant",
        "",
        extra=("updrs2_score", "updrs3_score", "pigd", "scopa", "rem", "moca", "SDMTOTAL", "VLTANIM", "HVLTRDLY"),
        qualification="Requires external composite construction and reference cutoffs.",
    ),
    Task(
        "S08",
        "subtype",
        "Data-driven progression phenotype",
        "",
        extra=("moca", "updrs2_score", "updrs3_score", "MSEADLG"),
        qualification="No frozen cross-release subtype labels.",
    ),
    Task(
        "S09",
        "subtype",
        "Baseline recorded RBD vs hyposmia enrollment",
        "subgroup",
        cohorts=(4,),
        labels=("RBD label", "Hyposmia label"),
    ),
    # Biomarkers
    classified(
        "B01",
        "biomarker",
        "PD baseline CSF SAA LBD-like positive vs negative",
        "CSFSAA",
        {0: "Negative", 1: "LBD-like positive"},
        qualification="Multiple assay generations across releases.",
    ),
    classified(
        "B02",
        "biomarker",
        "Prodromal baseline CSF SAA LBD-like positive vs negative",
        "CSFSAA",
        {0: "Negative", 1: "LBD-like positive"},
        cohorts=(4,),
        qualification="Withheld early prodromal results and assay changes.",
    ),
    Task("B03", "biomarker", "Baseline serum NfL in PD", "nfl_serum"),
    Task(
        "B04",
        "biomarker",
        "Two-year MoCA in baseline NfL subgroup",
        "moca",
        "M24",
        features=("age", "moca", "nfl_serum"),
        must_measure=("nfl_serum",),
    ),
    Task(
        "B05",
        "biomarker",
        "PD baseline recorded NSD stage",
        "NSD_STAGE",
        labels=("0", "1A", "1B", "2A", "2B", "3", "4", "5", "6", "Not NSD"),
        qualification="Staging definition changes and sparse categories require a dedicated design.",
    ),
    Task("B06", "biomarker", "Baseline bilateral striatal DaT ratio (MIAKAT)", "MIA_STRIATUM_BILAT"),
    Task(
        "B07",
        "biomarker",
        "PD baseline Lilly plasma p-tau217",
        "Lilly_ptau217p",
        extra=("Lilly_ptau217p_LLOD",),
        qualification="Short series and below-quantification handling required.",
    ),
    Task(
        "B08",
        "biomarker",
        "PD baseline C2N plasma A-beta42/40 ratio",
        "C2N_AB4240",
        extra=("C2N_AB_Contaminant",),
        qualification="Short series and assay QC handling required.",
    ),
    # Prodromal outcomes
    continuous("P01", "Two-year MoCA in prodromal cohort", "moca", "prodromal", (4,)),
    continuous("P02", "Two-year MDS-UPDRS III (OFF or untreated) in prodromal cohort", "updrs3_score", "prodromal", (4,)),
    Task(
        "P03",
        "prodromal",
        "Prodromal month-24 recorded PD diagnosis proxy",
        "PRIMDIAG",
        "M24",
        (4,),
        labels=("Nonmanifest", "PD"),
        risk=(("PRIMDIAG", "in", (17, 23, 24, 25)),),
        qualification="Proxy diagnosis without independent adjudication or conversion date.",
    ),
    Task(
        "P04",
        "prodromal",
        "Prodromal month-24 recorded PD or DLB diagnosis proxy",
        "PRIMDIAG",
        "M24",
        (4,),
        labels=("Nonmanifest", "PD or DLB"),
        risk=(("PRIMDIAG", "in", (17, 23, 24, 25)),),
        qualification="Proxy diagnosis without independent adjudication or conversion date.",
    ),
    # Unidentifiable candidate endpoints audited for completeness
    conditional("X01", "function", "First fall", "", "Curated cut lacks a common fall endpoint.", extra=("pm_wb_any",)),
    conditional("X02", "function", "Recurrent falls", "", "Curated cut lacks a common recurrent fall item.", extra=("pm_wb_any",)),
    conditional("X03", "function", "Freezing of gait", "", "Walking/balance composite cannot resolve freezing separately.", extra=("pm_wb_any",)),
    conditional("X04", "function", "New dyskinesia", "", "MDS-UPDRS IV total cannot resolve the specific dyskinesia item.", extra=("updrs4_score",)),
    conditional("X05", "function", "New motor fluctuations", "", "MDS-UPDRS IV total cannot resolve the motor fluctuation items.", extra=("updrs4_score",)),
    conditional("X06", "function", "First dopaminergic treatment", "", "Recorded PDTRTMNT includes DBS; medication onset alone is unavailable.", extra=("PDTRTMNT", "LEDD")),
)


def extract_filename(version: str) -> str:
    if version == "20230612":
        return "PPMI_Curated_Data_Cut_Public_20230612_rev.csv"
    return f"PPMI_Curated_Data_Cut_Public_{version}.xlsx"


def resolve_data_dir(cli_dir: str | Path | None = None) -> Path:
    """Resolve curated PPMI data directory: CLI --data-dir > PPMI_DATA_DIR > local data/raw."""
    candidates: list[Path] = []
    if cli_dir is not None:
        base = Path(cli_dir)
        candidates.extend([base, base / "curated"])
    env = os.environ.get("PPMI_DATA_DIR")
    if env:
        base = Path(env)
        candidates.extend([base, base / "curated"])
    candidates.extend([PROJECT / "data" / "raw", PROJECT / "data" / "raw" / "curated"])

    marker = extract_filename(REFERENCE)
    for path in candidates:
        if (path / marker).is_file():
            return path
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Curated PPMI extract '{marker}' not found. Pass --data-dir, set PPMI_DATA_DIR, "
        f"or place the 11 official curated files in data/raw/. Checked: {searched}"
    )


def snapshot_month(version: str) -> int:
    return int(version[:4]) * 12 + int(version[4:6]) - 2


def calendar_month(index: int) -> str:
    year, month_idx = divmod(index, 12)
    return f"{year:04d}-{month_idx + 1:02d}"


def key(value) -> str:
    return str(value).strip().removesuffix(".0")


def number(value) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def parse_month(value) -> int | None:
    if isinstance(value, (date, datetime)):
        return value.year * 12 + value.month - 1
    raw = str(value).strip()
    for fmt in ("%m/%Y", "%Y-%m", "%b%Y", "%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.year * 12 + parsed.month - 1
        except ValueError:
            pass
    return None


def agree(values):
    items = list(values)
    return items[0] if items and None not in items and len(set(items)) == 1 else None


def text(value) -> str | None:
    cleaned = "" if value is None else str(value).strip()
    return None if cleaned.upper() in {"", ".", "NA", "N/A", "NAN", "NULL", "UNKNOWN", "-"} else cleaned


def normalized(field: str, raw):
    raw = text(raw)
    if raw is None:
        return None
    if field in {"visit_date", "Death_Date"}:
        return parse_month(raw)
    if field == "PATNO":
        return key(raw)
    if field in {"EVENT_ID", "subgroup", "analytic_subgroup", "NSD_STAGE"}:
        return raw.upper()
    value = number(raw)
    lower, upper = BOUNDS.get(field, (0, math.inf))
    return value if value is not None and lower <= value <= upper else None


def selected_rows(data_dir: Path, version: str, columns: set[str]):
    path = data_dir / extract_filename(version)
    if version == "20230612":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = set(reader.fieldnames or ())
            yield headers
            selected = headers & columns
            for row in reader:
                yield {f: normalized(f, row[f]) for f in selected if text(row[f]) is not None}
        return

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = next(s for s in workbook if re.fullmatch(r"\d{8}", s.title))
        header = next(sheet.values)
        yield set(header) - {None}
        lookup = {get_column_letter(i): f for i, f in enumerate(header, 1) if f in columns}
        with ZipFile(path) as archive, archive.open(sheet._worksheet_path) as stream:
            for _, element in iterparse(stream, events=("end",)):
                if not element.tag.endswith("}row"):
                    continue
                if element.get("r") != "1":
                    row = {}
                    for cell in element:
                        col_name = lookup.get(cell.attrib.get("r", "").rstrip("0123456789"))
                        if col_name is None:
                            continue
                        value = cell.find("{*}v")
                        kind = cell.get("t")
                        if value is not None:
                            raw = sheet._shared_strings[int(value.text)] if kind == "s" else value.text
                        elif kind == "inlineStr":
                            raw = "".join(part.text or "" for part in cell.iterfind("{*}is//{*}t"))
                        else:
                            continue
                        row[col_name] = normalized(col_name, raw)
                    yield row
                element.clear()
    finally:
        workbook.close()


def combine(rows: list[dict]) -> dict:
    fields = set().union(*(row.keys() for row in rows))
    return {field: agree(row.get(field) for row in rows) for field in fields}


def people_for(data_dir: Path, version: str, columns: set[str]):
    """Parse one curated extract under the unified V[0-9]+ scheduled follow-up rule."""
    cutoff = snapshot_month(version)
    rows = selected_rows(data_dir, version, columns)
    headers = next(rows)
    by_person = defaultdict(list)
    for row in rows:
        identity = row.get("PATNO")
        if identity is not None:
            by_person[identity].append(row)
    people = {}
    for identity, records in by_person.items():
        cohort = agree(row.get("COHORT") for row in records)
        baselines = [row for row in records if row.get("EVENT_ID") == "BL"]
        base_month = agree(row.get("visit_date") for row in baselines)
        if cohort not in COHORT or base_month is None or base_month > cutoff:
            continue
        months = defaultdict(list)
        for row in records:
            visit = row.get("visit_date")
            if (
                visit is not None
                and base_month < visit <= cutoff
                and re.fullmatch(r"V[0-9]+", row.get("EVENT_ID") or "")
            ):
                months[visit].append(row)
        index_month = min(
            (m for m in months if base_month + 21 <= m <= base_month + 27),
            key=lambda m: (abs(m - base_month - 24), m),
            default=None,
        )
        future = combine(months[index_month]) if index_month is not None else {}
        people[identity] = (
            int(cohort),
            combine(baselines),
            future,
            int(base_month),
            int(max(months, default=base_month)),
        )
    return headers, people, cutoff


def map_label(task: Task, value):
    if value is None:
        return None
    field = task.target
    if field == "COHORT":
        return COHORT.get(value)
    if field == "cogstate":
        if task.id == "T1":
            return {1: "NC", 2: "MCI or dementia", 3: "MCI or dementia"}.get(value)
        if task.id == "K05":
            return {1: "NC or MCI", 2: "NC or MCI", 3: "Dementia"}.get(value)
        return COGNITION.get(value)
    if field == "td_pigd":
        return MOTOR.get(value)
    if field == "CSFSAA":
        return {0: "Negative", 1: "LBD-like positive"}.get(value)
    if field == "PRIMDIAG":
        if value in (17, 23, 24, 25):
            return "Nonmanifest"
        return {1: "PD"}.get(value) if task.id == "P03" else {1: "PD or DLB", 5: "PD or DLB"}.get(value)
    if field == "NSD_STAGE":
        return {"NOT NSD": "Not NSD"}.get(value, value)
    if field == "subgroup":
        if task.id == "S09":
            return {"RBD": "RBD label", "HYPOSMIA": "Hyposmia label"}.get(value)
        if task.id == "S05":
            rbd = bool(re.search(r"\bRBD\b", value))
            hyposmia = "HYPOSMIA" in value
            return {(True, False): "RBD", (False, True): "Hyposmia", (True, True): "Both"}.get((rbd, hyposmia))
        genes = set(GENES.findall(value))
        if not genes:
            return "Sporadic-type" if "SPORADIC" in value or value == "PD NORMOSMIC" else None
        if task.id == "S03":
            return "Genetic"
        if genes == {"LRRK2"}:
            return "LRRK2"
        return "GBA" if genes in ({"GBA"}, {"GBA1"}) else None
    if field == "analytic_subgroup":
        return {"A1": "PSG-RBD", "A2": "PSG-RBD", "C": "Hyposmia alone"}.get(value)
    if task.risk:
        if field == "NHY":
            return NO_YES[value >= 3]
        if field == "MSEADLG":
            return NO_YES[value < 80]
        return NO_YES.get(value)
    if field == "TMT_B" and value <= 0:
        return None
    return value


def eligible(cohort: int, baseline: dict, task: Task) -> bool:
    if cohort not in task.cohorts:
        return False
    for field, relation, threshold in task.risk:
        value = baseline.get(field)
        if value is None or (relation == "eq" and value != threshold):
            return False
        if (relation == "lt" and not value < threshold) or (relation == "ge" and not value >= threshold):
            return False
        if relation == "in" and value not in threshold:
            return False
    return True


def collect(task: Task, headers: set[str], people: dict, cutoff: int):
    present = task.columns
    if not task.target:
        complete = sum(
            cohort in task.cohorts and all(bl.get(f) is not None for f in task.extra)
            for cohort, bl, _, _, _ in people.values()
        )
        return "definition_unavailable", set(), {}, complete
    if not present <= headers:
        return "column_unavailable", set(), {}, None
    base_ids, records = set(), {}
    for pid, (cohort, baseline, followup, baseline_month, last_visit) in people.items():
        if not eligible(cohort, baseline, task):
            continue
        base_ids.add(pid)
        if task.mode != "BL" and baseline_month + (24 if task.mode == "DEATH" else 27) > cutoff:
            continue
        prediction_row = followup if task.mode == "VISIT24" else baseline
        observed_features = {field for field in task.features if prediction_row.get(field) is not None}
        if not observed_features or not set(task.must_measure) <= observed_features:
            continue
        target_row = baseline if task.mode == "BL" else followup
        if task.mode == "DEATH":
            dated = baseline.get("Death_Date")
            dead = baseline.get("Death_Status") == 1
            value = None
            if dead and dated is not None and baseline_month < dated <= cutoff:
                value = task.labels[dated <= baseline_month + 24]
            elif not dead and last_visit >= baseline_month + 24:
                value = task.labels[0]
        else:
            value = map_label(task, target_row.get(task.target))
        if value is not None and (not task.labels or value in task.labels):
            records[pid] = (value, len(observed_features) == len(task.features))
    return "available", base_ids, records, None


def unified_reserve(versions: tuple[str, ...], cohorts: dict[str, dict[str, int]]) -> tuple[set[str], dict]:
    """Single-pass 30% deterministic stratified evaluation reservation with no legacy overrides."""
    reference = versions[-1]
    still_common = set(cohorts[reference])
    earliest_common: dict[str, str] = {}
    for version in reversed(versions):
        still_common &= set(cohorts[version])
        for identity in still_common:
            earliest_common[identity] = version

    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    for identity, first in earliest_common.items():
        if first != reference:
            groups[(cohorts[reference][identity], first)].append(identity)

    reserved: set[str] = set()
    stratum_summary: dict[str, dict[str, int]] = {}
    for (cohort_code, first_cut), identities in sorted(groups.items()):
        ordered = sorted(
            identities,
            key=lambda p: (hashlib.sha256(f"{RESERVE_SEED}:{p}".encode("utf-8")).hexdigest(), p),
        )
        take = math.floor(RESERVE_FRACTION * len(ordered) + 0.5)
        reserved.update(ordered[:take])
        stratum_summary[f"{COHORT[cohort_code]}:{first_cut}"] = {"eligible": len(ordered), "reserved": take}

    eligible_total = sum(len(v) for v in groups.values())
    by_cohort = {
        COHORT[c]: {
            "eligible": sum(1 for pid, first in earliest_common.items() if first != reference and cohorts[reference][pid] == c),
            "reserved": sum(1 for pid in reserved if cohorts[reference][pid] == c),
        }
        for c in sorted(COHORT)
    }
    metadata = {
        "seed": RESERVE_SEED,
        "fraction": RESERVE_FRACTION,
        "eligible_pool": eligible_total,
        "total_reserved": len(reserved),
        "by_cohort": by_cohort,
        "strata": stratum_summary,
    }
    return reserved, metadata


def counts(records: dict, ids, task: Task) -> dict[str, int]:
    grouped = Counter(records[p][0] for p in ids) if task.labels else {}
    return {name: grouped.get(name, 0) for name in task.labels}


def comparison(task: Task, cuts: tuple[str, ...], observations: dict, reserved: set[str]) -> dict:
    data = [observations[v] for v in cuts]
    if not all(status == "available" for status, _, _, _ in data):
        return dict(
            common_ref_n=0,
            strict_common_n=0,
            eval_n=0,
            eval_complete_n=0,
            eval_classes=counts({}, (), task),
            common_ref_classes=counts({}, (), task),
            strict_common_classes=counts({}, (), task),
            min_dev_n=0,
            min_dev_class=0 if task.labels else None,
            dev={},
        )
    common = set.intersection(*(base for _, base, _, _ in data))
    latest = data[-1][2]
    reference = common & set(latest)
    strict = reference.intersection(*(set(valid) for _, _, valid, _ in data))
    evaluation = reference & reserved
    dev = {}
    for version, (_, _, valid, _) in zip(cuts, data):
        ids = set(valid) - reserved
        dev[version] = (len(ids), counts(valid, ids, task))
    assert not any(evaluation & (set(valid) - reserved) for _, _, valid, _ in data)
    return dict(
        common_ref_n=len(reference),
        strict_common_n=len(strict),
        eval_n=len(evaluation),
        eval_complete_n=sum(latest[p][1] for p in evaluation),
        eval_classes=counts(latest, evaluation, task),
        common_ref_classes=counts(latest, reference, task),
        strict_common_classes=counts(latest, strict, task),
        min_dev_n=min(n for n, _ in dev.values()),
        min_dev_class=min((n for _, cls in dev.values() for n in cls.values()), default=None),
        dev=dev,
    )


def supported(task: Task, stats: dict) -> bool:
    if task.labels:
        return min(stats["eval_classes"].values()) >= 10 and stats["min_dev_class"] >= 20
    return stats["eval_n"] >= 50 and stats["min_dev_n"] >= 100
