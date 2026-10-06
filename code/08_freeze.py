"""Freeze the model specification and unseal the test origins.

Run once, after the dev-stage results have been reviewed and any design
changes logged in docs/decisions.md. Writes docs/FROZEN_SPEC.json with a hash
of the feature sets and tuning grids; 04_models.py test refuses to run if the
spec has changed since. Commit the file: the commit timestamp is the record
that the test set was scored after the freeze.
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from common import FINAL, FREEZE_FILE, ROOT

spec = importlib.util.spec_from_file_location("m", ROOT/"code"/"04_models.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

if FREEZE_FILE.exists() and "--force" not in sys.argv:
    sys.exit("Already frozen: {}".format(FREEZE_FILE.read_text()))
cols = pd.read_parquet(FINAL/"panel.parquet").columns
fs = m.feature_sets(cols)
FREEZE_FILE.write_text(json.dumps({
    "spec_hash": m.spec_hash(fs),
    "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "features": fs, "kappa": m.KAPPA, "lr_grid": m.LR_GRID, "gb_grid": m.GB_GRID,
}, indent=2))
print("frozen:", FREEZE_FILE)
