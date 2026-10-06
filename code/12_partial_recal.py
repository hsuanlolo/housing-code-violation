"""SECONDARY (post-freeze). Recalibrate on partially observed outcome windows.

Full-window rolling recalibration (11_recalibration.py) is always a year
behind: at origin t the latest *closed* 6-month window belongs to t-12m. The
origin t-6m has a window that is ~5 months observed by the information cutoff
t-30d. This script uses it, chain-ladder style:

  completion factor  cf(t) = rate_full(L) / rate_partial(L)
                     where L = t-12m and rate_partial(L) counts only events
                     that would have been visible at L's analogous cutoff
                     (inspected and approved by L + 6m - 30d)
  target rate        r*(t) = cf(t) * rate_partial(L'),  L' = t-6m, visible by t-30d

Strategies (on top of the Platt map fitted on L, as in roll_platt):
  part_int   intercept shift so the mean prediction on L' rows equals r*(t)
  part_dist  part_int, then shrunken per-district offsets toward
             cf(t) * district partial counts at L' (LAMBDA as in script 11)

  anchor     added after the first run showed that the drift is input inflation
             (event rate flat, mean prediction rising): shift the intercept so the
             mean prediction over t's own in-district rows equals r*(t). Uses t's
             features, which are known at t, and no outcome from t's window.
  anchor_dist  anchor, plus the part_dist district offsets

Compared with frozen / roll_platt / roll_dist from 11_recalibration.py.
Outputs: out/tables/sec_partial_*.csv, out/figures/sec_partial_by_origin.png
"""
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, FINAL, HORIZON_MONTHS, INFO_LAG_DAYS, OUT, TABLES, add_months, origins, ts


def load(name):
    spec = importlib.util.spec_from_file_location(name.split(".")[0].replace("-", "_"), "code/" + name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R11 = load("11_recalibration.py")
BP, EV = R11.BP, R11.EV
MODELS = R11.MODELS
LAG = pd.Timedelta(days=INFO_LAG_DAYS)


def partial_labels(v, o, cutoff):
    """1 if a Class C inspected in [o, cutoff) was approved by the cutoff."""
    o = ts(o)
    w = v[(v["inspectiondate"] >= o) & (v["inspectiondate"] < cutoff) & (v["approveddate"] <= cutoff)]
    return set(w["buildingid"])


def solve_shift(z, target_rate, iters=30):
    d = 0.0
    for _ in range(iters):
        q = R11.sigmoid(z + d)
        d += (target_rate - q.mean()) * len(q) / max((q * (1 - q)).sum(), 1e-9)
    return d


def main():
    panel = R11.build_extra_origins(pd.read_parquet(FINAL/"panel.parquet"))
    score = R11.fit_and_score(panel)
    v = BP.load_violations()

    out, diag = [], []
    for t in origins("test"):
        T = ts(t)
        L, Lp = add_months(T, -12), add_months(T, -6)
        cut_p, cut_L = T - LAG, add_months(L, HORIZON_MONTHS) - LAG
        sL = score[score["origin"] == L.strftime("%Y-%m-%d")].copy()
        sP = score[score["origin"] == Lp.strftime("%Y-%m-%d")].copy()
        sT = score[score["origin"] == t].copy()
        sL["y_part"] = sL["buildingid"].isin(partial_labels(v, L, cut_L)).astype(int)
        sP["y_part"] = sP["buildingid"].isin(partial_labels(v, Lp, cut_p)).astype(int)
        inL, inP = sL[sL["spatial_holdout"] == 0], sP[sP["spatial_holdout"] == 0]
        cf = inL["y"].mean() / inL["y_part"].mean()
        target = cf * inP["y_part"].mean()
        diag.append(dict(origin=t, L=L.date(), L_partial=Lp.date(), months_observed=round((cut_p - Lp).days / 30.44, 1),
                         completion_factor=cf, est_full_rate_Lp=target, actual_full_rate_Lp=inP["y"].mean(),
                         full_rate_L=inL["y"].mean()))
        r = sT[["buildingid", "origin", "cd", "spatial_holdout", "y"]].copy()
        for m in MODELS:
            f = R11.platt(inL[m].values, inL["y"].values)
            d = solve_shift(R11.logit(f(inP[m].values)), target)
            r[m + "|part_int"] = R11.sigmoid(R11.logit(f(sT[m].values)) + d)
            # District offsets toward completed partial counts at L' (all districts).
            zP = R11.logit(f(sP[m].values)) + d
            off = {}
            for cd, g in pd.DataFrame({"z": zP, "y": sP["y_part"].values * cf, "cd": sP["cd"].values}) \
                           .dropna(subset=["cd"]).groupby("cd"):
                dd = 0.0
                for _ in range(10):
                    q = R11.sigmoid(g["z"] + dd)
                    dd += (g["y"].sum() - q.sum() - R11.LAMBDA * dd) / ((q * (1 - q)).sum() + R11.LAMBDA)
                off[cd] = dd
            shift = sT["cd"].map(off).fillna(0).values
            r[m + "|part_dist"] = R11.sigmoid(R11.logit(r[m + "|part_int"].values) + shift)
            zT = R11.logit(f(sT[m].values))
            inT = (sT["spatial_holdout"] == 0).values
            dA = solve_shift(zT[inT], target)
            r[m + "|anchor"] = R11.sigmoid(zT + dA)
            r[m + "|anchor_dist"] = R11.sigmoid(zT + dA + shift)
        out.append(r)
    df = pd.concat(out, ignore_index=True)
    diag = pd.DataFrame(diag)
    diag.to_csv(TABLES/"sec_partial_diagnostics.csv", index=False)
    print(diag.round(4).to_string(index=False))

    prev = pd.read_parquet(OUT/"predictions_sec_recal.parquet")
    keep = ["buildingid", "origin"] + [c for c in prev.columns if "|" in c]
    df = df.merge(prev[keep], on=["buildingid", "origin"], how="left")
    df.to_parquet(OUT/"predictions_sec_partial.parquet", index=False)

    strats = ["frozen", "roll_platt", "roll_dist", "part_int", "part_dist", "anchor", "anchor_dist"]
    rows, byo = [], []
    for m in MODELS:
        for s in strats:
            c = "{}|{}".format(m, s)
            r = dict(model=m, strategy=s, brier=np.mean((df[c] - df["y"]) ** 2), o_e=df["y"].mean() / df[c].mean())
            r.update(R11.district_summary(df, c))
            rows.append(r)
            for o, g in df[df["spatial_holdout"] == 0].groupby("origin"):
                byo.append(dict(model=m, strategy=s, origin=o, o_e=g["y"].mean() / g[c].mean()))
    ov, byo = pd.DataFrame(rows), pd.DataFrame(byo)
    ov.to_csv(TABLES/"sec_partial_overall.csv", index=False)
    byo.to_csv(TABLES/"sec_partial_by_origin.csv", index=False)
    print(ov.round(4).to_string(index=False))
    print(byo.pivot_table(index=["model", "strategy"], columns="origin", values="o_e").round(3).to_string())

    con = []
    for m in MODELS:
        for a, b in [("part_int", "roll_platt"), ("part_dist", "roll_dist"), ("anchor", "roll_platt"),
                     ("anchor_dist", "roll_dist"), ("anchor_dist", "frozen")]:
            for split in ("in_district", "held_out_district"):
                d = df[df["spatial_holdout"] == int(split == "held_out_district")]
                est, lo, hi = EV.paired_bootstrap(d, "{}|{}".format(m, a), "{}|{}".format(m, b))
                con.append(dict(model=m, contrast="{} - {}".format(a, b), split=split, d_brier=est, ci_lo=lo, ci_hi=hi))
    con = pd.DataFrame(con)
    con.to_csv(TABLES/"sec_partial_contrasts.csv", index=False)
    print(con[con["model"].isin(["M2c", "M2e"])].to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    sty = {"frozen": ("--", 1.2), "roll_platt": ("-", 1.2), "part_int": (":", 1.6), "anchor": ("-", 2.4)}
    for s in sty:
        d = byo[(byo["model"] == "M2e") & (byo["strategy"] == s)]
        ax.plot(pd.to_datetime(d["origin"]), d["o_e"], sty[s][0], lw=sty[s][1], marker="o", ms=4,
                label=s.replace("_", "-"))
    ax.axhline(1, color="#999", lw=1)
    ax.set_ylabel("Observed / expected (M2e, in-district)")
    ax.set_xticks(pd.to_datetime(origins("test")))
    ax.set_xticklabels([o[:7] for o in origins("test")])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES/"sec_partial_by_origin.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
