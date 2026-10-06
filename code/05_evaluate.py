"""Evaluate predictions: overall, by spatial hold-out, and by community district.

Primary metric: Brier score (lower is better), with Brier skill score against
the training base rate. Secondary: precision/recall at a fixed 10% review
budget, AUC, calibration intercept/slope, expected calibration error.

Model differences use a paired bootstrap that resamples buildings (all of a
building's origins move together), so repeated observations of one building
cannot shrink the interval.

Usage: python code/05_evaluate.py [dev|test]
Writes out/tables/<stage>_*.csv and out/figures/<stage>_reliability.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import FIGURES, OUT, SEED, TABLES, TOP_K_SHARE

PROB_MODELS = ["M1", "M2a", "M2b", "M2c", "M2d", "M3"]
B = 500


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def top_k_mask(score, share=TOP_K_SHARE):
    n = max(1, int(round(len(score) * share)))
    # Random tie-breaking: the count baseline has many ties.
    rng = np.random.RandomState(SEED)
    order = np.lexsort((rng.rand(len(score)), -np.asarray(score)))
    m = np.zeros(len(score), bool)
    m[order[:n]] = True
    return m


def calib_slope(p, y):
    if y.min() == y.max():
        return np.nan, np.nan
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def ece(p, y, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((d["n"] * (d["p"] - d["y"]).abs()).sum() / d["n"].sum())


def metrics(df, col, base_rate, prob=True):
    y, s = df["y"].values, df[col].values
    m = top_k_mask(s)
    out = {"n": len(y), "events": int(y.sum()), "event_rate": y.mean(),
           "precision_at_10": y[m].mean(), "recall_at_10": y[m].sum() / max(1, y.sum()),
           "auc": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan}
    if prob:
        bs = np.mean((s - y) ** 2)
        ref = np.mean((base_rate - y) ** 2)
        a, b = calib_slope(s, y)
        out.update({"brier": bs, "brier_skill": 1 - bs / ref, "mean_pred": s.mean(),
                    "o_e_ratio": y.mean() / s.mean(), "cal_intercept": a, "cal_slope": b,
                    "ece": ece(s, y)})
    return out


def paired_bootstrap(df, a, b):
    """Brier(a) - Brier(b), resampling buildings. Negative favours a."""
    g = df.assign(da=(df[a] - df["y"]) ** 2, db=(df[b] - df["y"]) ** 2) \
          .groupby("buildingid")[["da", "db"]].agg(["sum", "count"])
    sa, sb, n = g[("da", "sum")].values, g[("db", "sum")].values, g[("da", "count")].values
    rng = np.random.RandomState(SEED)
    k = len(n)
    reps = []
    for _ in range(B):
        i = rng.randint(0, k, k)
        reps.append((sa[i].sum() - sb[i].sum()) / n[i].sum())
    est = (sa.sum() - sb.sum()) / n.sum()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return est, lo, hi


def reliability_plot(df, cols, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    for c in cols:
        q = pd.qcut(df[c], 10, labels=False, duplicates="drop")
        d = df.groupby(q).agg(p=(c, "mean"), y=("y", "mean"))
        ax.plot(d["p"], d["y"], marker="o", ms=3, lw=1.2, label=c)
    lim = max(df[cols].quantile(0.99).max(), df["y"].mean() * 3)
    ax.set_xlim(0, min(1, lim * 1.1)); ax.set_ylim(0, min(1, lim * 1.1))
    ax.set_xlabel("Mean predicted probability (decile)")
    ax.set_ylabel("Observed share with new Class C violation")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def district_plot(bys, path):
    """Observed/expected by district against district event rate, one panel per model (H4)."""
    fig, axes = plt.subplots(1, len(bys), figsize=(5 * len(bys), 4), sharey=True)
    for ax, (model, by) in zip(np.atleast_1d(axes), bys.items()):
        for held, mk, lab in [(0, "o", "in-district"), (1, "^", "held-out district")]:
            d = by[by["held_out"] == held]
            ax.scatter(d["event_rate"], d["o_e_ratio"], s=d["n"] / 40, marker=mk,
                       alpha=0.7, label=lab, edgecolor="white", linewidth=0.5)
        ax.axhline(1, color="#999", lw=1, ls="--")
        for _, r in by[(by["o_e_ratio"] < 0.6) | (by["o_e_ratio"] > 1.1)].iterrows():
            ax.annotate(str(int(r["cd"])), (r["event_rate"], r["o_e_ratio"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points", color="#444")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("District event rate")
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Observed / expected")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main(stage):
    df = pd.read_parquet(OUT/"predictions_{}.parquet".format(stage))
    meta = json.loads((OUT/"model_meta_{}.json".format(stage)).read_text())
    base = meta["base_rate_train"]
    cal = [c for c in df.columns if c.endswith("_cal")]
    probs = PROB_MODELS + cal

    # 1. Overall, split by spatial hold-out status.
    rows = []
    for split, d in [("all", df), ("in_district", df[df["spatial_holdout"] == 0]),
                     ("held_out_district", df[df["spatial_holdout"] == 1])]:
        for c in probs + ["M1r"]:
            rows.append(dict(split=split, model=c, **metrics(d, c, base, prob=(c != "M1r"))))
    overall = pd.DataFrame(rows)
    overall.to_csv(TABLES/"{}_overall.csv".format(stage), index=False)
    show = overall[overall["split"] != "all"][["split", "model", "n", "event_rate", "brier",
                                               "brier_skill", "precision_at_10", "recall_at_10",
                                               "auc", "o_e_ratio", "cal_slope"]]
    print(show.round(4).to_string(index=False))

    # 2. Paired model contrasts (RQ1-RQ4). H1-H3 on in-district rows; the
    #    district-context contrast also on held-out districts, where it matters.
    pairs = [("M2b", "M2a", "H1: timing/type vs counts", "in_district"),
             ("M2a", "M1", "logistic counts vs binned baseline", "in_district"),
             ("M2c", "M2b", "H2: + complaints", "in_district"),
             ("M3", "M2c", "H3: boosting vs logistic", "in_district"),
             ("M2d", "M2c", "H4b: + district context", "in_district"),
             ("M2d", "M2c", "H4b: + district context", "held_out_district")]
    con = []
    for a, b, lab, split in pairs:
        d = df[df["spatial_holdout"] == int(split == "held_out_district")]
        est, lo, hi = paired_bootstrap(d, a, b)
        con.append(dict(contrast="{} - {}".format(a, b), hypothesis=lab, split=split,
                        d_brier=est, ci_lo=lo, ci_hi=hi))
    con = pd.DataFrame(con)
    con.to_csv(TABLES/"{}_contrasts.csv".format(stage), index=False)
    print(con.to_string(index=False))

    # 3. Community-district reliability (RQ4), with and without district
    #    context. Recall uses each model's citywide top-10% cut, as a product
    #    would apply it.
    bys = {}
    for model in ("M2c", "M2d"):
        by = by_district(df, model)
        by.to_csv(TABLES/"{}_by_cd_{}.csv".format(stage, model), index=False)
        print("O/E across districts ({}): median {:.2f}, IQR {:.2f}-{:.2f}, range {:.2f}-{:.2f}; "
              "recall at citywide cut {:.2f}-{:.2f}; Spearman(O/E, event rate) {:.2f}".format(
                  model, by["o_e_ratio"].median(), *by["o_e_ratio"].quantile([.25, .75]),
                  by["o_e_ratio"].min(), by["o_e_ratio"].max(),
                  by["recall_citywide_cut"].min(), by["recall_citywide_cut"].max(),
                  by[["o_e_ratio", "event_rate"]].corr("spearman").iloc[0, 1]))
        bys[model] = by
    district_plot(bys, FIGURES/"{}_district_calibration.png".format(stage))
    ind = df[df["spatial_holdout"] == 0]
    reliability_plot(ind, ["M1", "M2b", "M2c", "M2d", "M3"], FIGURES/"{}_reliability.png".format(stage),
                     "Reliability, {} origin(s), in-district".format(stage))


def by_district(df, model):
    cut = np.sort(df[model].values)[::-1][max(1, int(len(df) * TOP_K_SHARE)) - 1]
    d = df.dropna(subset=["cd"]).assign(flag=lambda x: x[model] >= cut)
    by = d.groupby("cd").apply(lambda g: pd.Series({
        "n": len(g), "events": g["y"].sum(), "event_rate": g["y"].mean(),
        "mean_pred": g[model].mean(), "o_e_ratio": g["y"].mean() / g[model].mean(),
        "brier": np.mean((g[model] - g["y"]) ** 2),
        "recall_citywide_cut": g.loc[g["flag"], "y"].sum() / max(1, g["y"].sum()),
        "share_flagged": g["flag"].mean(),
        "c365_mean": g["c365"].mean(),
        "held_out": int(g["spatial_holdout"].iat[0])}))
    return by.reset_index()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
