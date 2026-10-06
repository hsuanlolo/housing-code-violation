"""Pre-registered prospective evaluation (docs/preregistration.md).

  python code/15_prereg.py register           write docs/prereg/PREREG_SPEC.json (hash of code + spec)
  python code/15_prereg.py predict  <origin>  score origin t; writes predictions + sha256, never outcomes
  python code/15_prereg.py evaluate <origin>  compute outcomes and the registered tests

Guards
  * predict/evaluate refuse to run unless the code and spec hash match the registration.
  * predict drops the outcome columns for origin t before anything is saved or printed.
  * evaluate refuses unless the predictions file matches the sha256 committed by
    predict, and unless the data were downloaded on or after t + 6 months + 60 days.
  * --dev runs the whole procedure on a development origin (<= 2025-07-01) to
    test the code; it bypasses the guards and writes to out/prereg_dev/.

Procedure for origin t (all fixed in the registration; nothing is tuned):
  training origins   2020-01-01 ... t - 18 months (half-yearly), in-district rows only
  calibration origin L  = t - 12 months (closed window)
  anchoring origin   L' = t - 6 months, observed to t - 30 days, completed by the
                     chain-ladder factor estimated at L
  product            M2d raw-count inputs -> Platt(L) -> anchor to completed rate
                     over t's in-district rows -> shrunken district offsets toward
                     completed partial counts at L' (LAMBDA = 100)
  comparator         M2c -> Platt(L)
  also scored        M2a, M2b, M2c, M2e (Platt(L)); M2d_rk (rank inputs, Platt(L))
"""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import FINAL, HORIZON_MONTHS, INFO_LAG_DAYS, OUT, RAW, ROOT, TOP_K_SHARE, add_months, ts

PREREG_DIR = ROOT/"docs"/"prereg"
SPEC_FILE = PREREG_DIR/"PREREG_SPEC.json"
CODE_FILES = ["common.py", "03_build_panel.py", "04_models.py", "05_evaluate.py", "10_acs_context.py",
              "11_recalibration.py", "12_partial_recal.py", "14_rank_features.py", "15_prereg.py"]
CONFIRMATORY = {"2027-01-01": "A-primary", "2027-07-01": "A-replication",
                "2026-01-01": "B-sealed-backtest", "2026-07-01": "B-sealed-backtest"}
LAMBDA = 100.0
B = 1000
SEED = 20261006


def load(name):
    spec = importlib.util.spec_from_file_location("m" + name.split("_")[0], ROOT/"code"/name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((ROOT/"code"/f).read_bytes())
    return h.hexdigest()


def fixed_params():
    test = json.loads((OUT/"model_meta_test.json").read_text())
    acs = json.loads((OUT/"model_meta_sec_acs.json").read_text())
    rk = json.loads((OUT/"model_meta_sec_rank.json").read_text())
    return {"M2a": test["params"]["M2a"], "M2b": test["params"]["M2b"], "M2c": test["params"]["M2c"],
            "M2d": test["params"]["M2d"], "M2e": acs["C"], "M2d_rk": rk["params"]["M2d_rk"],
            "held_out_cds": test["held_out_cds"]}


def register():
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    if SPEC_FILE.exists() and "--force" not in sys.argv:
        sys.exit("Already registered: " + SPEC_FILE.read_text()[:300])
    spec = {"registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_sha256": code_hash(), "code_files": CODE_FILES, "params": fixed_params(),
            "lambda": LAMBDA, "bootstrap_B": B, "seed": SEED, "origins": CONFIRMATORY,
            "doc_sha256": hashlib.sha256((ROOT/"docs"/"preregistration.md").read_bytes()).hexdigest()}
    SPEC_FILE.write_text(json.dumps(spec, indent=2))
    print("registered", spec["code_sha256"][:16])


def check_registration():
    if not SPEC_FILE.exists():
        sys.exit("Not registered. Run: python code/15_prereg.py register")
    spec = json.loads(SPEC_FILE.read_text())
    if spec["code_sha256"] != code_hash():
        sys.exit("Code changed since registration; deviations must be logged and re-registered.")
    if spec["doc_sha256"] != hashlib.sha256((ROOT/"docs"/"preregistration.md").read_bytes()).hexdigest():
        sys.exit("preregistration.md changed since registration.")
    return spec


# --------------------------------------------------------------------------- build
def half_years(start, end):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="6MS") if d.month in (1, 7)]


def build(origins_needed, BP, ACSM, RK, params):
    v, c = BP.load_violations(), BP.load_complaints()
    dev = c[c["month"] < "2023-07-01"]
    top = dev.groupby("major_category")["n_problems"].sum().nlargest(BP.TOP_COMPLAINT_CATS).index.tolist()
    geo = pd.read_csv(FINAL/"buildings_geo.csv", dtype={"buildingid": str})[["buildingid", "cd"]]
    parts = [BP.build_origin(v, c, o, top) for o in origins_needed]
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.fillna({x: 0 for x in panel.columns if x.startswith(("c365_cat_", "comp365_cat_"))})
    panel = BP.add_district_context(panel.merge(geo, on="buildingid", how="left"))
    panel["spatial_holdout"] = panel["cd"].isin(set(params["held_out_cds"])).astype(int)
    return panel, v


def predict(t, dev=False):
    spec = None if dev else check_registration()
    if not dev and t not in CONFIRMATORY:
        sys.exit("Not a registered origin: " + t)
    if dev and t > "2025-07-01":
        sys.exit("--dev is only for development origins (<= 2025-07-01)")
    params = spec["params"] if spec else fixed_params()
    BP, M, ACSM, RK, R11, R12 = (load("03_build_panel.py"), load("04_models.py"), load("10_acs_context.py"),
                                 load("14_rank_features.py"), load("11_recalibration.py"), load("12_partial_recal.py"))
    T = ts(t)
    L, Lp = add_months(T, -12), add_months(T, -6)
    train_o = half_years("2020-01-01", add_months(T, -18))
    need = train_o + [L.strftime("%Y-%m-%d"), Lp.strftime("%Y-%m-%d"), t]
    panel, v = build(need, BP, ACSM, RK, params)
    panel = ACSM.attach_acs(panel, (panel["spatial_holdout"] == 0) & panel["origin"].isin(train_o))
    # Outcomes of t are never kept: drop them before anything else touches the rows.
    is_t = panel["origin"] == t
    panel.loc[is_t, ["y", "y_count"]] = np.nan

    fs = M.feature_sets(panel.columns)
    fs["M2e"] = fs["M2c"] + ACSM.ACS + ["acs_missing"]
    RK.RANKED = RK.rank_cols(sorted(set(fs["M2d"])))
    panel = RK.add_ranks(panel, RK.RANKED)
    fs["M2d_rk"] = RK.to_rank(fs["M2d"])

    inside = panel["spatial_holdout"] == 0
    fit = panel[inside & panel["origin"].isin(train_o)]
    sL = panel[panel["origin"] == need[-3]].copy()
    sP = panel[panel["origin"] == need[-2]].copy()
    sT = panel[is_t].copy()
    cut_L, cut_p = add_months(L, HORIZON_MONTHS) - pd.Timedelta(days=INFO_LAG_DAYS), T - pd.Timedelta(days=INFO_LAG_DAYS)
    sL["y_part"] = sL["buildingid"].isin(R12.partial_labels(v, L, cut_L)).astype(int)
    sP["y_part"] = sP["buildingid"].isin(R12.partial_labels(v, Lp, cut_p)).astype(int)
    inL, inP = sL[sL["spatial_holdout"] == 0], sP[sP["spatial_holdout"] == 0]
    cf = inL["y"].mean() / inL["y_part"].mean()
    target = cf * inP["y_part"].mean()

    out = sT[["buildingid", "origin", "cd", "spatial_holdout"]].copy()
    for name in ["M2a", "M2b", "M2c", "M2d", "M2e", "M2d_rk"]:
        cols = fs[name]
        mk = RK.make_lr_rank if name == "M2d_rk" else (ACSM.make_lr if name == "M2e" else M.make_lr)
        model = mk(params[name], cols).fit(fit[cols].values, fit["y"].values)
        f = R11.platt(model.predict_proba(inL[cols].values)[:, 1], inL["y"].values)
        out[name] = f(model.predict_proba(sT[cols].values)[:, 1])
        if name == "M2d":
            zT = R11.logit(out[name].values)
            d = R12.solve_shift(zT[(sT["spatial_holdout"] == 0).values], target)
            zP = R11.logit(f(model.predict_proba(sP[cols].values)[:, 1])) + d
            off = {}
            for cd, g in pd.DataFrame({"z": zP, "y": sP["y_part"].values * cf, "cd": sP["cd"].values}) \
                           .dropna(subset=["cd"]).groupby("cd"):
                dd = 0.0
                for _ in range(10):
                    q = R11.sigmoid(g["z"] + dd)
                    dd += (g["y"].sum() - q.sum() - LAMBDA * dd) / ((q * (1 - q)).sum() + LAMBDA)
                off[cd] = dd
            out["product"] = R11.sigmoid(zT + d + sT["cd"].map(off).fillna(0).values)
    out["flag_district"] = load("05_evaluate.py").flag(out.assign(y=0), "product", "district_relative")
    meta = {"origin": t, "train_origins": train_o, "calib_origin": need[-3], "anchor_origin": need[-2],
            "completion_factor": cf, "anchor_target": target, "n": len(out),
            "data_fetch": json.loads((RAW/"fetch_log.json").read_text())}

    d = (OUT/"prereg_dev") if dev else (OUT/"prereg")
    d.mkdir(parents=True, exist_ok=True)
    path = d/"pred_{}.parquet".format(t)
    out.to_parquet(path, index=False)
    (d/"pred_{}.meta.json".format(t)).write_text(json.dumps(meta, indent=2, default=str))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not dev:
        (PREREG_DIR/"pred_{}.sha256".format(t)).write_text(digest + "\n")
        print("Commit docs/prereg/pred_{}.sha256 now (before evaluating).".format(t))
    print("scored", len(out), "buildings at", t, "| anchor target {:.4f} | sha256 {}".format(target, digest[:16]))


# --------------------------------------------------------------------------- evaluate
def boot(df, stat, rng):
    """Resample buildings. One origin per evaluation, so one row per building."""
    assert df["buildingid"].is_unique
    n = len(df)
    return np.array([stat(df.iloc[rng.randint(0, n, n)]) for _ in range(B)])


def dist_err(df, col):
    by = df.dropna(subset=["cd"]).groupby("cd").agg(y=("y", "mean"), p=(col, "mean"))
    return float(((by["y"] / by["p"]) - 1).abs().mean())


def evaluate(t, dev=False):
    spec = None if dev else check_registration()
    d = (OUT/"prereg_dev") if dev else (OUT/"prereg")
    path = d/"pred_{}.parquet".format(t)
    if not dev:
        reg = (PREREG_DIR/"pred_{}.sha256".format(t)).read_text().strip()
        if hashlib.sha256(path.read_bytes()).hexdigest() != reg:
            sys.exit("Predictions file does not match the committed sha256.")
        fetched = pd.Timestamp(json.loads((RAW/"fetch_log.json").read_text())["violations"]["fetched_utc"]).tz_localize(None)
        if fetched < add_months(ts(t), HORIZON_MONTHS) + pd.Timedelta(days=60):
            sys.exit("Data downloaded {} is too early to evaluate {}.".format(fetched.date(), t))
    pred = pd.read_parquet(path)
    BP = load("03_build_panel.py")
    v = BP.load_violations()
    T = ts(t)
    end = add_months(T, HORIZON_MONTHS)
    hit = set(v.loc[(v["inspectiondate"] >= T) & (v["inspectiondate"] < end), "buildingid"])
    df = pred.assign(y=pred["buildingid"].isin(hit).astype(int))
    rng = np.random.RandomState(spec["seed"] if spec else SEED)
    res = {"origin": t, "tier": CONFIRMATORY.get(t, "dev"), "n": len(df), "event_rate": df["y"].mean()}
    for split, g in (("in_district", df[df["spatial_holdout"] == 0]), ("held_out", df[df["spatial_holdout"] == 1]),
                     ("all", df)):
        for m in ["M2a", "M2b", "M2c", "M2d", "M2e", "M2d_rk", "product"]:
            res["{}|brier|{}".format(split, m)] = float(np.mean((g[m] - g["y"]) ** 2))
            res["{}|oe|{}".format(split, m)] = float(g["y"].mean() / g[m].mean())
        res["{}|dist_err|product".format(split)] = dist_err(g, "product")
        res["{}|dist_err|M2c".format(split)] = dist_err(g, "M2c")

    ind = df[df["spatial_holdout"] == 0]
    tests = {
        "P1_level": (lambda x: abs(x["y"].mean() / x["product"].mean() - 1) - abs(x["y"].mean() / x["M2c"].mean() - 1), ind),
        "P2_district": (lambda x: dist_err(x, "product") - dist_err(x, "M2c"), df),
        "P3_brier": (lambda x: np.mean((x["product"] - x["y"]) ** 2) - np.mean((x["M2c"] - x["y"]) ** 2), ind),
        "S1_H1": (lambda x: np.mean((x["M2b"] - x["y"]) ** 2) - np.mean((x["M2a"] - x["y"]) ** 2), ind),
        "S2_H2": (lambda x: np.mean((x["M2c"] - x["y"]) ** 2) - np.mean((x["M2b"] - x["y"]) ** 2), ind),
        "S3_rank": (lambda x: dist_err(x, "M2d_rk") - dist_err(x, "M2d"), df),
    }
    pvals = {}
    for k, (stat, data) in tests.items():
        est = stat(data)
        bs = boot(data, stat, rng)
        p = float((np.sum(bs >= 0) + 1) / (B + 1))           # one-sided: H_a is "< 0"
        pvals[k] = p
        res[k] = {"estimate": float(est), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "p_one_sided": p}
    # Holm across the three primary hypotheses.
    prim = sorted(["P1_level", "P2_district", "P3_brier"], key=lambda k: pvals[k])
    for i, k in enumerate(prim):
        res[k]["holm_threshold"] = 0.05 / (3 - i)
    stop = False
    for i, k in enumerate(prim):
        ok = (not stop) and pvals[k] <= 0.05 / (3 - i)
        res[k]["supported"] = bool(ok)
        stop = stop or not ok
    for k in ("S1_H1", "S2_H2", "S3_rank"):
        res[k]["supported"] = res[k]["p_one_sided"] <= 0.05

    EV = load("05_evaluate.py")
    for rule, col in (("citywide", None), ("district_relative", "flag_district")):
        f = EV.flag(df, "product", "citywide") if col is None else df[col].values.astype(bool)
        by = df.assign(f=f).dropna(subset=["cd"]).groupby("cd").apply(
            lambda g: g.loc[g["f"], "y"].sum() / max(1, g["y"].sum()))
        res["S4_" + rule] = {"recall": float(df["y"].values[f].sum() / df["y"].sum()),
                             "districts_lt10": int((by < 0.10).sum()), "min_recall": float(by.min())}
    cost = 1 - res["S4_district_relative"]["recall"] / res["S4_citywide"]["recall"]
    res["S4_supported"] = bool(res["S4_district_relative"]["districts_lt10"] == 0 and cost <= 0.20)
    res["S4_recall_cost"] = float(cost)

    (d/"eval_{}.json".format(t)).write_text(json.dumps(res, indent=2))
    for k in ("P1_level", "P2_district", "P3_brier", "S1_H1", "S2_H2", "S3_rank"):
        print("{:12s} est {:+.5f}  95% CI [{:+.5f}, {:+.5f}]  p {:.4f}  supported {}".format(
            k, res[k]["estimate"], *res[k]["ci95"], res[k]["p_one_sided"], res[k]["supported"]))
    print("S4 warning rule: lt10 {} -> {}, recall cost {:.1%}, supported {}".format(
        res["S4_citywide"]["districts_lt10"], res["S4_district_relative"]["districts_lt10"], cost, res["S4_supported"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    dev = "--dev" in sys.argv
    if cmd == "register":
        register()
    elif cmd == "predict":
        predict(sys.argv[2], dev)
    elif cmd == "evaluate":
        evaluate(sys.argv[2], dev)
