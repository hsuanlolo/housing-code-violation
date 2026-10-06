"""SECONDARY (post-freeze). Can scheduled recalibration keep probabilities honest under drift?

The frozen models were recalibrated once, on the 2023-01 origin, and their
calibration decayed through 2024-25 (paper Section 7.3). Here the model fits
are left untouched and only the mapping from score to probability is
refreshed, using the most recent origin whose outcome window had closed by the
information cutoff at t. With half-yearly origins and a 6-month horizon that
is always the origin 12 months earlier: L(2024-01) = 2023-01,
L(2024-07) = 2023-07, L(2025-01) = 2024-01, L(2025-07) = 2024-07.
(2023-07 was the unused buffer origin; it is built here for this purpose.)

Strategies, each applied to the frozen raw score of M2c, M2d, M3 and to M2e:
  raw         no recalibration
  frozen      Platt on 2023-01 (the pre-specified *_cal)
  roll_int    intercept shift on L(t): calibration-in-the-large only
  roll_platt  Platt (intercept + slope) on L(t)
  roll_dist   roll_platt, then a shrunken per-district intercept shift on L(t)

Recalibration data: in-district rows for the citywide mappings, so held-out
districts never inform anything citywide. roll_dist estimates each district's
offset from that district's own past outcomes at L(t), held-out districts
included: a deployed tool would know them. The model fits never see them.
Shrinkage LAMBDA is fixed a priori, not tuned.

Outputs: out/predictions_sec_recal.parquet, out/tables/sec_recal_*.csv,
         out/figures/sec_recal_by_origin.png
"""
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import FIGURES, FINAL, OUT, TABLES, add_months, origins

LAMBDA = 100.0        # pseudo-information for district offsets (fixed a priori)
EXTRA = ["2023-07-01"]
MODELS = ["M2c", "M2d", "M3", "M2e"]
STRATS = ["raw", "frozen", "roll_int", "roll_platt", "roll_dist"]


def load(name):
    spec = importlib.util.spec_from_file_location(name.split(".")[0].replace("-", "_"), "code/" + name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BP, M, EV, ACSM = (load("03_build_panel.py"), load("04_models.py"),
                   load("05_evaluate.py"), load("10_acs_context.py"))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def build_extra_origins(panel):
    v, c = BP.load_violations(), BP.load_complaints()
    dev = c[c["month"] < "2023-07-01"]
    top = dev.groupby("major_category")["n_problems"].sum().nlargest(BP.TOP_COMPLAINT_CATS).index.tolist()
    geo = pd.read_csv(FINAL/"buildings_geo.csv", dtype={"buildingid": str})[["buildingid", "cd"]]
    parts = []
    for t in EXTRA:
        x = BP.build_origin(v, c, t, top).merge(geo, on="buildingid", how="left")
        parts.append(BP.add_district_context(x))
        print("built", t, len(parts[-1]), "rows")
    extra = pd.concat(parts, ignore_index=True)
    extra = extra.reindex(columns=panel.columns).fillna(
        {k: 0 for k in panel.columns if k.startswith(("c365_cat_", "comp365_cat_"))})
    return pd.concat([panel, extra], ignore_index=True)


def fit_and_score(panel):
    """Re-fit the frozen spec exactly as `04_models.py test`, and M2e as 10_acs_context."""
    meta = json.loads((OUT/"model_meta_test.json").read_text())
    held = M.spatial_holdout(panel)
    panel["spatial_holdout"] = panel["cd"].isin(held).astype(int)
    train_mask = (panel["spatial_holdout"] == 0) & panel["origin"].isin(origins("train"))
    panel = ACSM.attach_acs(panel, train_mask)
    fs = M.feature_sets(panel.columns)
    fit = panel[(panel["spatial_holdout"] == 0) & panel["origin"].isin(origins("train") + origins("tune"))]
    score = panel[panel["origin"].isin(origins("calib") + EXTRA + origins("test"))].copy()
    for name in ["M2c", "M2d", "M3"]:
        cols, prm = fs[name], meta["params"][name]
        model = M.make_lr(prm, cols) if name.startswith("M2") else M.make_gb(prm)
        model.fit(fit[cols].values, fit["y"].values)
        score[name] = model.predict_proba(score[cols].values)[:, 1]
        print("fitted", name)
    acs_meta = json.loads((OUT/"model_meta_sec_acs.json").read_text())
    cols = fs["M2c"] + ACSM.ACS + ["acs_missing"]
    model = ACSM.make_lr(acs_meta["C"], cols).fit(fit[cols].values, fit["y"].values)
    score["M2e"] = model.predict_proba(score[cols].values)[:, 1]

    # The re-fit must reproduce the frozen test predictions.
    ref = pd.read_parquet(OUT/"predictions_sec_acs.parquet")
    chk = score.merge(ref[["buildingid", "origin", "M2c", "M2d", "M3", "M2e"]],
                      on=["buildingid", "origin"], suffixes=("", "_ref"))
    for name in MODELS:
        diff = (chk[name] - chk[name + "_ref"]).abs().max()
        print("max |refit - frozen| {}: {:.2e}".format(name, diff))
        assert diff < 1e-6, name
    return score


def platt(p_fit, y_fit):
    lr = LogisticRegression(C=1e6).fit(logit(p_fit).reshape(-1, 1), y_fit)
    return lambda p: lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def intercept_shift(p_fit, y_fit, iters=25):
    z, d = logit(p_fit), 0.0
    for _ in range(iters):     # Newton steps on the single offset
        q = sigmoid(z + d)
        d += (y_fit - q).sum() / max((q * (1 - q)).sum(), 1e-9)
    return lambda p: sigmoid(logit(p) + d)


def district_offsets(p_fit, y_fit, cd_fit, iters=10):
    df = pd.DataFrame({"z": logit(p_fit), "y": y_fit, "cd": cd_fit}).dropna(subset=["cd"])
    off = {}
    for cd, g in df.groupby("cd"):
        d = 0.0
        for _ in range(iters):
            q = sigmoid(g["z"] + d)
            # Newton step on the penalized log-likelihood: sum loglik - LAMBDA/2 * d^2
            d += ((g["y"] - q).sum() - LAMBDA * d) / ((q * (1 - q)).sum() + LAMBDA)
        off[cd] = d
    return off


def apply_strategies(score):
    out = []
    test = score[score["origin"].isin(origins("test"))]
    cal23 = score[(score["origin"] == "2023-01-01") & (score["spatial_holdout"] == 0)]
    for t, g in test.groupby("origin"):
        L = add_months(t, -12).strftime("%Y-%m-%d")
        prev_all = score[score["origin"] == L]
        prev = prev_all[prev_all["spatial_holdout"] == 0]
        r = g[["buildingid", "origin", "cd", "spatial_holdout", "y"]].copy()
        r["recal_origin"] = L
        for m in MODELS:
            r[m + "|raw"] = g[m].values
            r[m + "|frozen"] = platt(cal23[m].values, cal23["y"].values)(g[m].values)
            r[m + "|roll_int"] = intercept_shift(prev[m].values, prev["y"].values)(g[m].values)
            f = platt(prev[m].values, prev["y"].values)
            r[m + "|roll_platt"] = f(g[m].values)
            off = district_offsets(f(prev_all[m].values), prev_all["y"].values, prev_all["cd"].values)
            shift = g["cd"].map(off).fillna(0).values
            r[m + "|roll_dist"] = sigmoid(logit(r[m + "|roll_platt"].values) + shift)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def district_summary(df, col):
    by = df.dropna(subset=["cd"]).groupby("cd").agg(y=("y", "mean"), p=(col, "mean"),
                                                    held=("spatial_holdout", "first"))
    oe = by["y"] / by["p"]
    e = (oe - 1).abs()
    return dict(dist_mean_abs_err=e.mean(), dist_mean_abs_err_heldout=e[by["held"] == 1].mean(),
                dist_n_gt_20pct=int((e > 0.2).sum()),
                rho_oe_rate=pd.Series(oe).corr(by["y"], method="spearman"))


def main():
    panel = pd.read_parquet(FINAL/"panel.parquet")
    panel = build_extra_origins(panel)
    score = fit_and_score(panel)
    df = apply_strategies(score)
    df.to_parquet(OUT/"predictions_sec_recal.parquet", index=False)

    rows = []
    for split, d in [("in_district", df[df["spatial_holdout"] == 0]),
                     ("held_out_district", df[df["spatial_holdout"] == 1]), ("all", df)]:
        for m in MODELS:
            for s in STRATS:
                c = "{}|{}".format(m, s)
                r = dict(split=split, model=m, strategy=s, brier=np.mean((d[c] - d["y"]) ** 2),
                         o_e=d["y"].mean() / d[c].mean(), ece=EV.ece(d[c].values, d["y"].values))
                if split == "all":
                    r.update(district_summary(d, c))
                rows.append(r)
    ov = pd.DataFrame(rows)
    ov.to_csv(TABLES/"sec_recal_overall.csv", index=False)
    print(ov.round(4).to_string(index=False))

    byo = []
    for o, g in df[df["spatial_holdout"] == 0].groupby("origin"):
        for m in MODELS:
            for s in STRATS:
                c = "{}|{}".format(m, s)
                byo.append(dict(origin=o, model=m, strategy=s, o_e=g["y"].mean() / g[c].mean(),
                                brier=np.mean((g[c] - g["y"]) ** 2)))
    byo = pd.DataFrame(byo)
    byo.to_csv(TABLES/"sec_recal_by_origin.csv", index=False)
    print(byo[byo["model"].isin(["M2c", "M2e"])].pivot_table(
        index=["model", "strategy"], columns="origin", values="o_e").round(3).to_string())

    con = []
    for m in MODELS:
        for s in ["roll_int", "roll_platt", "roll_dist"]:
            for split in ("in_district", "held_out_district"):
                d = df[df["spatial_holdout"] == int(split == "held_out_district")]
                est, lo, hi = EV.paired_bootstrap(d, "{}|{}".format(m, s), "{}|frozen".format(m))
                con.append(dict(model=m, contrast="{} - frozen".format(s), split=split,
                                d_brier=est, ci_lo=lo, ci_hi=hi))
    con = pd.DataFrame(con)
    con.to_csv(TABLES/"sec_recal_contrasts.csv", index=False)
    print(con[con["model"].isin(["M2c", "M2e"])].to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    sty = {"raw": ":", "frozen": "--", "roll_int": "-.", "roll_platt": "-", "roll_dist": "-"}
    for s in STRATS:
        d = byo[(byo["model"] == "M2c") & (byo["strategy"] == s)]
        ax.plot(pd.to_datetime(d["origin"]), d["o_e"], sty[s], marker="o", ms=4,
                lw=2.2 if s == "roll_dist" else 1.3, label=s.replace("_", "-"))
    ax.axhline(1, color="#999", lw=1)
    ax.set_ylabel("Observed / expected (M2c, in-district)")
    ax.set_xticks(pd.to_datetime(origins("test")))
    ax.set_xticklabels([o[:7] for o in origins("test")])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES/"sec_recal_by_origin.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
