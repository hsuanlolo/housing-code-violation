"""SECONDARY (post-freeze). Sensitivity grid for the frozen test results.

Each variant changes one design choice, rebuilds the panel, and re-runs the
frozen test procedure unchanged (same held-out districts, tuning grids, fit
on train+tune, Platt recalibration on 2023-01, scoring on 2024-25):

  baseline        frozen design (must reproduce the frozen test numbers)
  lag0, lag60     information buffer 0 / 60 days instead of 30
  no_dismissed    drop violations whose *current* status is dismissed, from
                  features and outcome. Uses today's status, i.e. future
                  information -- shows how much dismissals matter, not a
                  deployable design
  stable_outcome  outcome excludes the two categories reclassified around 2019
                  (pests, doors/egress); features unchanged
  class_ab        adds Class A/B violation history to the M2b+ feature sets
  h3, h12         3- and 12-month outcome horizons

Usage: python code/13_sensitivity.py [variant ...]   (default: all)
Appends one row per variant to out/tables/sec_sensitivity.csv
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd

from common import FINAL, INFO_LAG_DAYS, RAW, TABLES, origins

STABLE = {"heat_water", "lead_paint", "structural", "mold_leak", "gas_electric",
          "detectors_guards", "plumbing", "sanitation", "other"}
AB = ["a365", "b365", "b90", "b365_months"]
VARIANTS = ["baseline", "lag0", "lag60", "no_dismissed", "stable_outcome", "class_ab", "h3", "h12"]
OUTFILE = TABLES/"sec_sensitivity.csv"


def load(name):
    spec = importlib.util.spec_from_file_location(name.split(".")[0].replace("-", "_") + str(time.time()).replace(".", ""),
                                                  "code/" + name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ab_features(ab, t, lag):
    cut = t - pd.Timedelta(days=lag)
    known = ab[ab["month"] + pd.offsets.MonthBegin(1) <= cut]
    f = pd.DataFrame(index=pd.Index(known["buildingid"].unique(), name="buildingid"))
    y1 = known[known["month"] >= t - pd.Timedelta(days=365)]
    q1 = known[known["month"] >= t - pd.Timedelta(days=90)]
    f["a365"] = y1[y1["class"] == "A"].groupby("buildingid")["n"].sum()
    f["b365"] = y1[y1["class"] == "B"].groupby("buildingid")["n"].sum()
    f["b90"] = q1[q1["class"] == "B"].groupby("buildingid")["n"].sum()
    f["b365_months"] = y1[y1["class"] == "B"].groupby("buildingid")["month"].nunique()
    return f.fillna(0)


def build_panel(variant):
    BP = load("03_build_panel.py")
    lag = {"lag0": 0, "lag60": 60}.get(variant, INFO_LAG_DAYS)
    BP.INFO_LAG_DAYS = lag
    BP.HORIZON_MONTHS = {"h3": 3, "h12": 12}.get(variant, BP.HORIZON_MONTHS)
    v, c = BP.load_violations(), BP.load_complaints()
    if variant == "no_dismissed":
        st = pd.read_csv(RAW/"violations_classC.csv", usecols=["violationid", "currentstatus"], dtype=str)
        dismissed = set(st.loc[st["currentstatus"] == "VIOLATION DISMISSED", "violationid"])
        v = v[~v["violationid"].isin(dismissed)]
    if variant == "stable_outcome":
        orig = BP.outcome
        BP.outcome = lambda vv, t: orig(vv[vv["category"].isin(STABLE)], t)
    dev = c[c["month"] < "2023-07-01"]
    top = dev.groupby("major_category")["n_problems"].sum().nlargest(BP.TOP_COMPLAINT_CATS).index.tolist()
    if variant == "class_ab":
        ab = pd.read_csv(RAW/"violations_ab_bym.csv", dtype={"buildingid": str})
        ab["month"] = pd.to_datetime(ab["month"])
        ab["n"] = ab["n"].astype(int)
    parts = []
    for t in origins():
        p = BP.build_origin(v, c, t, top)
        if variant == "class_ab":
            p = p.merge(ab_features(ab, pd.Timestamp(t), lag), on="buildingid", how="left")
            p[AB] = p[AB].fillna(0)
        parts.append(p)
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.fillna({x: 0 for x in panel.columns if x.startswith(("c365_cat_", "comp365_cat_"))})
    geo = pd.read_csv(FINAL/"buildings_geo.csv", dtype={"buildingid": str})[["buildingid", "cd"]]
    panel = panel.merge(geo, on="buildingid", how="left")
    return BP.add_district_context(panel)


def run_models(panel, variant, M):
    if variant == "class_ab":
        M.TIMING = M.TIMING + AB
    held = M.spatial_holdout(panel)
    panel["spatial_holdout"] = panel["cd"].isin(held).astype(int)
    fs = M.feature_sets(panel.columns)
    inside = panel[panel["spatial_holdout"] == 0]
    train = inside[inside["origin"].isin(origins("train"))]
    tune = inside[inside["origin"].isin(origins("tune"))]
    calib = inside[inside["origin"].isin(origins("calib"))]
    fit = pd.concat([train, tune])
    score = panel[panel["origin"].isin(origins("test"))].copy()
    preds = score[["buildingid", "origin", "cd", "spatial_holdout", "y", "c365", "c730"]].copy()
    m1 = M.fit_m1(fit)
    preds["M1"] = M.predict_m1(m1, score)
    preds["M1r"] = score["c365"] + score["c730"] / 1000.0
    for name, cols in fs.items():
        prm = M.tune(name, cols, train, tune)
        mdl = M.make_lr(prm, cols) if name.startswith("M2") else M.make_gb(prm)
        mdl.fit(fit[cols].values, fit["y"].values)
        preds[name] = mdl.predict_proba(score[cols].values)[:, 1]
        f, _ = M.recalibrate(mdl.predict_proba(calib[cols].values)[:, 1], calib["y"].values)
        preds[name + "_cal"] = f(preds[name].values)
    return preds, float(train["y"].mean())


def summarize(variant, df, base, EV):
    ind, ho = df[df["spatial_holdout"] == 0], df[df["spatial_holdout"] == 1]
    r = {"variant": variant, "n_test": len(df), "event_rate": df["y"].mean(), "train_rate": base}
    for m in ["M1", "M2a", "M2b", "M2c", "M2d", "M3"]:
        r["brier_" + m] = np.mean((ind[m] - ind["y"]) ** 2)
    for lab, a, b, d in [("H1", "M2b", "M2a", ind), ("H2", "M2c", "M2b", ind), ("H3", "M3", "M2c", ind),
                         ("H4b_in", "M2d", "M2c", ind), ("H4b_ho", "M2d", "M2c", ho)]:
        est, lo, hi = EV.paired_bootstrap(d, a, b)
        r[lab] = est
        r[lab + "_lo"], r[lab + "_hi"] = lo, hi
        r[lab + "_rel"] = est / np.mean((d[b] - d["y"]) ** 2)
    r["p10_M2c"] = EV.metrics(ind, "M2c", base)["precision_at_10"]
    r["p10_M1r"] = EV.metrics(ind, "M1r", base, prob=False)["precision_at_10"]
    oe = ind.groupby("origin").apply(lambda g: g["y"].mean() / g["M2c_cal"].mean())
    r["oe_cal_first"], r["oe_cal_last"] = oe.iloc[0], oe.iloc[-1]
    for m in ["M2c", "M2d"]:
        by = EV.by_district(df, m)
        r["dist_rho_" + m] = by[["o_e_ratio", "event_rate"]].corr("spearman").iloc[0, 1]
        r["dist_gt20_" + m] = int(((by["o_e_ratio"] - 1).abs() > 0.2).sum())
    for rule in ("citywide", "district_relative"):
        by = EV.by_district(df, "M2d", rule)
        f = EV.flag(df, "M2d", rule)
        r["recall_" + rule] = df["y"].values[f].sum() / df["y"].sum()
        r["lt10_" + rule] = int((by["recall_citywide_cut"] < 0.10).sum())
        r["minrecall_" + rule] = by["recall_citywide_cut"].min()
    return r


def main(variants):
    for variant in variants:
        t0 = time.time()
        M, EV = load("04_models.py"), load("05_evaluate.py")
        panel = build_panel(variant)
        df, base = run_models(panel, variant, M)
        row = summarize(variant, df, base, EV)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        out = pd.DataFrame([row])
        if OUTFILE.exists():
            prev = pd.read_csv(OUTFILE)
            out = pd.concat([prev[prev["variant"] != variant], out], ignore_index=True)
        out.to_csv(OUTFILE, index=False)
        print("== {} done in {} min: brier M2c {:.4f}, H1 {:+.4f}, H2 {:+.4f}, H3 {:+.4f}, H4b_ho {:+.4f}".format(
            variant, row["minutes"], row["brier_M2c"], row["H1"], row["H2"], row["H3"], row["H4b_ho"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or VARIANTS)
