"""Shared paths, study design constants and small helpers.

Every design choice that could be tuned after seeing results lives here, so a
diff of this file is a diff of the study design. Change it only with an entry
in docs/decisions.md.
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW, INTERIM, FINAL, OUT = (ROOT/"data"/"raw", ROOT/"data"/"interim",
                            ROOT/"data"/"final", ROOT/"out")
TABLES, FIGURES = OUT/"tables", OUT/"figures"
for d in (RAW, INTERIM, FINAL, TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# --- NYC Open Data (Socrata) -------------------------------------------------
SOCRATA = "https://data.cityofnewyork.us/resource"
DS_VIOLATIONS = "wvxf-dwi5"   # Housing Maintenance Code Violations
DS_COMPLAINTS = "ygpa-z7cr"   # Housing Maintenance Code Complaints and Problems
DS_BUILDINGS = "kj4p-ruqc"    # Buildings Subject to HPD Jurisdiction
HISTORY_START = "2017-01-01"  # 24-month lookback before the first origin + slack

# --- Prediction design -------------------------------------------------------
HORIZON_MONTHS = 6            # outcome window (t, t + 6 months]
LOOKBACK_MONTHS = 24          # eligibility: any record in [t - 24m, t)
INFO_LAG_DAYS = 30            # information buffer: only records approved <= t - 30d
# Origins are 1 Jan and 1 Jul. Roles are fixed before any model is fit.
ORIGINS = {
    "train": ["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01", "2022-01-01"],
    "tune": ["2022-07-01"],
    "calib": ["2023-01-01"],
    # 2023-07-01 deliberately unused: lets every development outcome window close
    # before the model freeze date (2023-12-31).
    "test": ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"],
}
ROLE_OF = {o: r for r, os in ORIGINS.items() for o in os}

# Spatial hold-out: whole community districts withheld from every stage.
SPATIAL_HOLDOUT_SHARE = 0.20
SEED = 20261006

# Precision/recall are reported at a fixed review budget.
TOP_K_SHARE = 0.10

# Test-set seal. 04_models.py refuses to score "test" origins unless this file
# exists; 08_freeze.py writes it with a hash of the frozen model spec.
FREEZE_FILE = ROOT/"docs"/"FROZEN_SPEC.json"


def origins(role=None):
    if role is None:
        return sorted(ROLE_OF)
    return list(ORIGINS[role])


def ts(s):
    return pd.Timestamp(s)


def add_months(t, m):
    return ts(t) + pd.DateOffset(months=m)
