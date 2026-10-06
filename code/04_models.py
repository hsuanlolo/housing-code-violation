"""Fit the three model layers and write out-of-sample predictions.

  M1   historical-count baseline: smoothed event rate by Class C count bin x half
  M1r  ranking-only baseline: past-year Class C count (ties broken by 2-year count)
  M2a  logistic: counts + season
  M2b  logistic: + timing, repetition, problem type, history length
  M2c  logistic: + complaints
  M2d  logistic: + district context known at t (added 2026-10-06, pre-freeze)
  M3   gradient boosting on the M2c inputs

The layers are compared, not chained: no model consumes another's output.

Stages
  dev   fit on train origins, tune on the tune origin, evaluate on the calib
        origin (2023-01), plus the spatial hold-out districts at every origin.
  test  refuses to run unless docs/FROZEN_SPEC.json exists and matches the
        current model spec. Re-fits on train+tune, recalibrates on calib, and
        scores the 2024-2025 test origins once.

Output: out/predictions_<stage>.parquet
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import (FINAL, FREEZE_FILE, HORIZON_MONTHS, INFO_LAG_DAYS, LOOKBACK_MONTHS, ORIGINS,
                    OUT, PRODUCT_MODEL, SEED, SPATIAL_HOLDOUT_SHARE, TOP_K_SHARE, WARNING_RULE,
                    origins)

KAPPA = 20  # M1 smoothing strength (pseudo-observations at the overall rate)

COUNTS = ["c365", "c90"]
TIMING = ["days_since_c", "never_c", "c730", "c365_days", "c365_months", "trend",
          "c365_rentimp", "history_months"]
CD_CONTEXT = ["cd_share_c365", "cd_share_c90", "cd_mean_comp365"]  # see 03_build_panel.py
COMPLAINTS_BASE = ["comp90", "comp365", "comp730", "comp365_emerg", "comp365_months",
                   "comp_trend"]


def feature_sets(cols):
    cat = sorted(c for c in cols if c.startswith("c365_cat_"))
    ccat = sorted(c for c in cols if c.startswith("comp365_cat_"))
    m2a = COUNTS + ["half_jul"]
    m2b = m2a + TIMING + cat
    m2c = m2b + COMPLAINTS_BASE + ccat
    m2d = m2c + CD_CONTEXT
    return {"M2a": m2a, "M2b": m2b, "M2c": m2c, "M2d": m2d, "M3": m2c}


def spec_hash(fs):
    spec = {"kappa": KAPPA, "features": fs, "lr_C": LR_GRID, "gb_grid": GB_GRID,
            "design": {"horizon": HORIZON_MONTHS, "lag": INFO_LAG_DAYS, "lookback": LOOKBACK_MONTHS,
                       "origins": ORIGINS, "holdout": SPATIAL_HOLDOUT_SHARE, "seed": SEED,
                       "top_k": TOP_K_SHARE, "warning_rule": WARNING_RULE,
                       "product_model": PRODUCT_MODEL}}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


# --- M1 ----------------------------------------------------------------------
def m1_bin(c365):
    return pd.cut(c365, [-1, 0, 1, 4, np.inf], labels=["0", "1", "2-4", "5+"]).astype(str)


def fit_m1(train):
    pbar = train["y"].mean()
    g = train.assign(b=m1_bin(train["c365"])).groupby(["b", "half_jul"])["y"].agg(["sum", "count"])
    g["p"] = (g["sum"] + KAPPA * pbar) / (g["count"] + KAPPA)
    return g["p"].to_dict(), pbar


def predict_m1(model, df):
    table, pbar = model
    keys = zip(m1_bin(df["c365"]), df["half_jul"])
    return np.array([table.get(k, pbar) for k in keys])


# --- M2 / M3 -----------------------------------------------------------------
LR_GRID = [0.01, 0.1, 1.0]
GB_GRID = [dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=200)
           for lr in (0.05, 0.1) for leaves in (15, 31)]

# Heavy-tailed counts enter the logistic model as log(1 + x); days-since as log.
def _log1p_nonneg(X):
    return np.log1p(np.clip(X, 0, None))


def make_lr(C, cols):
    tf = FunctionTransformer(lambda X: np.column_stack([
        _log1p_nonneg(X[:, i])
        if c not in ("trend", "comp_trend", "half_jul", "never_c") and not c.startswith("cd_")
        else X[:, i] for i, c in enumerate(cols)]))
    return make_pipeline(tf, StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def make_gb(params):
    return HistGradientBoostingClassifier(max_iter=500, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20,
                                          random_state=SEED, **params)


def tune(name, cols, fit_df, val_df):
    grid = LR_GRID if name.startswith("M2") else GB_GRID
    best = None
    for g in grid:
        m = make_lr(g, cols) if name.startswith("M2") else make_gb(g)
        m.fit(fit_df[cols].values, fit_df["y"].values)
        b = brier_score_loss(val_df["y"], m.predict_proba(val_df[cols].values)[:, 1])
        print("    {} {} brier={:.5f}".format(name, g, b))
        if best is None or b < best[0]:
            best = (b, g)
    return best[1]


def recalibrate(p_cal, y_cal):
    """Platt-style logistic recalibration on the logit of the raw score."""
    z = np.log(np.clip(p_cal, 1e-6, 1 - 1e-6) / (1 - np.clip(p_cal, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y_cal)

    def f(p):
        zz = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return lr.predict_proba(zz.reshape(-1, 1))[:, 1]
    return f, (float(lr.intercept_[0]), float(lr.coef_[0, 0]))


def spatial_holdout(panel):
    cds = np.sort(panel["cd"].dropna().unique())
    rng = np.random.RandomState(SEED)
    k = int(round(len(cds) * SPATIAL_HOLDOUT_SHARE))
    return set(rng.choice(cds, size=k, replace=False))


def main(stage):
    panel = pd.read_parquet(FINAL/"panel.parquet")
    fs = feature_sets(panel.columns)
    held = spatial_holdout(panel)
    panel["spatial_holdout"] = panel["cd"].isin(held).astype(int)
    # Buildings without a district cannot be assigned to either side; they stay
    # in the temporal analysis and are excluded from the spatial comparison.
    print("held-out community districts:", sorted(int(x) for x in held))

    inside = panel[panel["spatial_holdout"] == 0]
    train = inside[inside["origin"].isin(origins("train"))]
    tune_df = inside[inside["origin"].isin(origins("tune"))]
    calib = inside[inside["origin"].isin(origins("calib"))]

    if stage == "test":
        if not FREEZE_FILE.exists():
            sys.exit("Test set is sealed. Run code/08_freeze.py after dev review.")
        frozen = json.loads(FREEZE_FILE.read_text())
        if frozen["spec_hash"] != spec_hash(fs):
            sys.exit("Model spec changed since freeze ({} != {}).".format(
                frozen["spec_hash"], spec_hash(fs)))
        score_origins = origins("test")
    else:
        score_origins = origins("calib")

    # Score every building at the scoring origins. Held-out districts never
    # entered fitting, so in-district vs held-out at the same origin isolates
    # spatial transfer from temporal drift.
    score = panel[panel["origin"].isin(score_origins)].copy()

    preds = score[["buildingid", "origin", "role", "cd", "spatial_holdout", "y", "y_count",
                   "c365", "c730"]].copy()
    meta = {"stage": stage, "spec_hash": spec_hash(fs), "held_out_cds": sorted(int(x) for x in held),
            "n_train": len(train), "n_tune": len(tune_df), "n_calib": len(calib),
            "base_rate_train": float(train["y"].mean()), "params": {}, "recal": {}}

    fit_final = pd.concat([train, tune_df]) if stage == "test" else train

    m1 = fit_m1(fit_final)
    preds["M1"] = predict_m1(m1, score)
    preds["M1r"] = score["c365"] + score["c730"] / 1000.0  # ranking score only

    for name, cols in fs.items():
        print("  fitting", name, len(cols), "features")
        params = tune(name, cols, train, tune_df)
        meta["params"][name] = params
        model = make_lr(params, cols) if name.startswith("M2") else make_gb(params)
        model.fit(fit_final[cols].values, fit_final["y"].values)
        preds[name] = model.predict_proba(score[cols].values)[:, 1]
        if name.startswith("M2"):
            lr = model.steps[-1][1]
            meta.setdefault("coef", {})[name] = dict(zip(cols, map(float, lr.coef_[0])))
        if stage == "test":
            f, ab = recalibrate(model.predict_proba(calib[cols].values)[:, 1], calib["y"].values)
            preds[name + "_cal"] = f(preds[name].values)
            meta["recal"][name] = ab

    preds.to_parquet(OUT/"predictions_{}.parquet".format(stage), index=False)
    (OUT/"model_meta_{}.json".format(stage)).write_text(json.dumps(meta, indent=2))
    print("wrote", len(preds), "scored rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
