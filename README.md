# Prediction Error Across Curated Releases and Chronological Training Sizes in 39 PPMI Tasks

Analysis and figure-generation code evaluating prediction error across 11 archived official curated releases and chronologically ordered training-size growth in 39 Parkinson's Precision Medicine Initiative (PPMI) clinical tabular tasks.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Paper:** Jing Wang et al. *Prediction error across curated releases and chronological training sizes in 39 PPMI tasks*. (Author list to be finalized prior to journal submission.)

## Overview

This pipeline evaluates 39 low-dimensional clinical prediction tasks (16 binary classification, 2 multinomial classification, and 21 continuous regression tasks across seven clinical and biomarker families) using two paired trajectories on a fixed evaluation reference per task. Across 4,788 participants present in at least two contiguous official curated extracts ending in August 2026, a single stratified deterministic 30% reservation (1,437 identities) defines the evaluation pool, and all 25 longitudinal tasks use a unified scheduled follow-up visit filter (`V01`, `V02`, etc., within months 21--27). Trajectory A fits L2-penalized logistic regression ($C=1$) or ridge regression ($\alpha=1$) on each official curated extract (409 task--extract estimates across 5--11 extracts). Trajectory B grows nested training prefixes within the August 2026 extract ordered by backward cross-extract verification (exact on clinical variables and outcomes, with a 0.01-year tolerance on exported age), using 1,000 nested within-snapshot permutations (seed `20261121`) whenever a tier splits a tied snapshot group.

## Data Access and Setup

No raw PPMI tables, participant-level derivatives, predictions, or fitted model checkpoints are distributed in this repository.

- **PPMI Curated Data Cuts (11 archived releases, 12 June 2023 to 10 August 2026; retrieved 22 October 2026)**: Obtain via the [PPMI data portal](https://www.ppmi-info.org/access-data-specimens/download-data) (RRID:SCR_006431) under the PPMI Data Use Agreement.

Place the 11 official curated files in `data/raw/` (or pass `--data-dir` / set `PPMI_DATA_DIR`). See [data/README.md](data/README.md) for the exact file list.

```bash
export PPMI_DATA_DIR=/path/to/ppmi/curated
# PowerShell: $env:PPMI_DATA_DIR = '/path/to/ppmi/curated'
```

## Environment and Usage

Analyses were executed in Python 3.14.6 with dependencies recorded in [requirements.txt](requirements.txt) and [environment_snapshot.json](environment_snapshot.json).

```bash
python -m pip install -r requirements.txt

# 1. Run 10-repetition pilot audit (partition, window-closure, and scikit-learn checks)
python run_unified_dual_trajectories.py --pilot --workers 2

# 2. Run full 1,000-repetition dual-trajectory evaluation (409 release points + 231 tiers)
python run_unified_dual_trajectories.py --workers 4

# 3. Generate summary tables and trajectory figures from aggregate CSV outputs
python build_unified_manuscript_assets.py
```

Run the synthetic unit tests without private PPMI data:

```bash
python -m unittest discover -s tests -t .
```

Aggregate CSV/JSON tables, LaTeX summary tables, and trajectory PDFs are written locally under `outputs/`.

## Repository Structure

```text
ppmi-data-cut-prediction/
├── README.md                              # Overview and reproduction guide
├── LICENSE                                # MIT license
├── CITATION.cff                           # Citation metadata
├── requirements.txt                       # Pinned Python dependencies
├── environment_snapshot.json              # Recorded runtime package versions
├── ppmi_task_registry.py                  # Curated extract parser, 39-task registry, unified reserve
├── run_unified_dual_trajectories.py       # Pilot audit and 1,000-round dual-trajectory runner
├── build_unified_manuscript_assets.py     # Aggregate summary tables and trajectory figures
├── tests/
│   ├── __init__.py
│   └── test_unified_dual_trajectories.py  # Pure-synthetic unit tests
└── data/
    └── README.md                          # Official PPMI file list and directory setup
```

## License and Contact

Released under the [MIT License](LICENSE). Official PPMI data remain subject to the PPMI Data Use Agreement.

Correspondence: Jing Wang (`wangjing@xynu.edu.cn`).
