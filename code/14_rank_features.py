"""SECONDARY (post-freeze, development for the 2026+ pre-registration).

Rank-normalized inputs: build inflation-robustness into the model itself.

Sections 7.3 and 7.7 showed that the drift is input inflation: record counts
per building rose ~40% while the event rate stayed flat, so models trained on
raw counts over-predict later. Here every enforcement-derived input is
replaced by its percentile rank among in-scope buildings *at the same origin*
(ties averaged). A building in the top 5% of recent Class C counts maps to
0.95 whether the city is recording 2 or 3 per building. Inputs that do not
inflate (season, history length, never-violated flag, ACS) are kept as they are.

  M2c_rk, M2d_rk, M2e_rk, M3_rk  =  rank-normalized versions of M2c, M2d, M2e, M3

Fitting mirrors the frozen test stage (held-out districts, tuning on 2022-07,
fit on train+tune, Platt on 2023-01) and is scored on 2024-25, which is now
development data: it has been used by every secondary analysis.

Calibration strategies compared for both input types:
  raw     model output
  cal     Platt on 2023-01 (as frozen)
  anchor  intercept shift so the mean prediction over the origin's own
          in-district rows equals the completed recent event rate
          (out/tables/sec_partial_diagnostics.csv; no outcome from t's window)

Outputs: out/tables/sec_rank_*.csv, out/figures/sec_rank_by_origin.png
"""
import importlib.util
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import FIGURES, FINAL, OUT, TABLES, origins

KEEP_RAW = ("half_jul", "never_c", "history_months", "acs_missing")


def load(name):
    spec = importlib.util.spec_from_file_location(name.split(".")[0].replace("-", "_"), "code/" + name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M, EV, ACSM = load("04_models.py"), load("05_evaluate.py"), load("10_acs_context.py")


def rank_cols(cols):
    return [c for c in cols if c not in KEEP_RAW and not c.startswith("acs_")]


def add_ranks(panel, cols):
    """Percentile rank of each enforcement-derived input within its origin (features only)."""
    g = panel.groupby("origin")
    for c in cols:
        panel["rk_" + c] = g[c].rank(pct=True, method="average")
    return panel


def to_rank(cols):
    return ["rk_" + c if c in RANKED else c for c in cols]


def make_lr_rank(C, cols):
    def tf(X):
        return np.column_stack([np.log1p(np.clip(X[:, i], 0, None)) if c == "history_months" else X[:, i]
                                for i, c in enumerate(cols)])
    return make_pipeline(FunctionTransformer(tf), StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def tune_fit(name, cols, train, tune, fit):
    is_lr = not name.startswith("M3")
    grid = M.LR_GRID if is_lr else M.GB_GRID
    best = None
    for g in grid:
        m = make_lr_rank(g, cols) if is_lr else M.make_gb(g)
        m.fit(train[cols].values, train["y"].values)
        b = brier_score_loss(tune["y"], m.predict_proba(tune[cols].values)[:, 1])
        if best is None or b < best[0]:
            best = (b, g)
    m = make_lr_rank(best[1], cols) if is_lr else M.make_gb(best[1])
    return m.fit(fit[cols].values, fit["y"].values), best[1]


def anchor(df, col, targets):
    out = np.empty(len(df))
    for o, idx in df.groupby("origin").groups.items():
        g = df.loc[idx]
        z = EV.logit(g[col].values)
        inside = (g["spatial_holdout"] == 0).values
        d = 0.0
        for _ in range(30):
            q = 1 / (1 + np.exp(-(z[inside] + d)))
            d += (targets[o] - q.mean()) * inside.sum() / (q * (1 - q)).sum()
        out[df.index.get_indexer(idx)] = 1 / (1 + np.exp(-(z + d)))
    return out


RANKED = []


def main():
    global RANKED
    panel = pd.read_parquet(FINAL/"panel.parquet")
    held = M.spatial_holdout(panel)
    panel["spatial_holdout"] = panel["cd"].isin(held).astype(int)
    train_mask = (panel["spatial_holdout"] == 0) & panel["origin"].isin(origins("train"))
    panel = ACSM.attach_acs(panel, train_mask)
    fs = M.feature_sets(panel.columns)
    fs["M2e"] = fs["M2c"] + ACSM.ACS + ["acs_missing"]
    RANKED = rank_cols(sorted(set(fs["M2d"])))
    panel = add_ranks(panel, RANKED)

    inside = panel[panel["spatial_holdout"] == 0]
    train = inside[inside["origin"].isin(origins("train"))]
    tune = inside[inside["origin"].isin(origins("tune"))]
    calib = inside[inside["origin"].isin(origins("calib"))]
    fit = pd.concat([train, tune])
    test = panel[panel["origin"].isin(origins("test"))].copy()

    ref = pd.read_parquet(OUT/"predictions_sec_acs.parquet")
    df = test[["buildingid", "origin", "cd", "spatial_holdout", "y", "c365"]].merge(
        ref[["buildingid", "origin", "M2c", "M2d", "M3", "M2e", "M2c_cal", "M2d_cal", "M3_cal", "M2e_cal"]],
        on=["buildingid", "origin"], how="left").set_index(test.index)
    meta = {"ranked_inputs": RANKED, "params": {}}
    for name in ["M2c", "M2d", "M2e", "M3"]:
        cols = to_rank(fs[name])
        model, prm = tune_fit(name, cols, train, tune, fit)
        meta["params"][name + "_rk"] = prm
        df[name + "_rk"] = model.predict_proba(test[cols].values)[:, 1]
        f, _ = M.recalibrate(model.predict_proba(calib[cols].values)[:, 1], calib["y"].values)
        df[name + "_rk_cal"] = f(df[name + "_rk"].values)
        print("fitted", name + "_rk", prm)
    df = df.reset_index(drop=True)

    diag = pd.read_csv(TABLES/"sec_partial_diagnostics.csv")
    targets = dict(zip(diag["origin"], diag["est_full_rate_Lp"]))
    base = ["M2c", "M2d", "M2e", "M3"]
    for m in base + [b + "_rk" for b in base]:
        df[m + "_anc"] = anchor(df, m, targets)
    df.to_parquet(OUT/"predictions_sec_rank.parquet", index=False)
    (OUT/"model_meta_sec_rank.json").write_text(json.dumps(meta, indent=2))

    rows, byo = [], []
    for b in base:
        for kind, m in (("raw counts", b), ("rank", b + "_rk")):
            for strat, col in (("raw", m), ("cal", m + "_cal"), ("anchor", m + "_anc")):
                ind = df[df["spatial_holdout"] == 0]
                r = dict(model=b, inputs=kind, strategy=strat,
                         brier_in=np.mean((ind[col] - ind["y"]) ** 2),
                         brier_ho=np.mean((df.loc[df["spatial_holdout"] == 1, col] - df.loc[df["spatial_holdout"] == 1, "y"]) ** 2),
                         auc=EV.roc_auc_score(ind["y"], ind[col]), o_e=df["y"].mean() / df[col].mean())
                oe = ind.groupby("origin").apply(lambda g: g["y"].mean() / g[col].mean())
                r["oe_first"], r["oe_last"] = oe.iloc[0], oe.iloc[-1]
                r["oe_range_by_origin"] = oe.max() - oe.min()
                by = EV.by_district(df, col)
                e = (by["o_e_ratio"] - 1).abs()
                r.update(dist_mean_abs_err=e.mean(), dist_mean_abs_err_ho=e[by["held_out"] == 1].mean(),
                         dist_gt20=int((e > 0.2).sum()),
                         rho_oe_rate=by[["o_e_ratio", "event_rate"]].corr("spearman").iloc[0, 1])
                rows.append(r)
                for o, v in oe.items():
                    byo.append(dict(model=b, inputs=kind, strategy=strat, origin=o, o_e=v))
    ov, byo = pd.DataFrame(rows), pd.DataFrame(byo)
    ov.to_csv(TABLES/"sec_rank_overall.csv", index=False)
    byo.to_csv(TABLES/"sec_rank_by_origin.csv", index=False)
    print(ov.round(4).to_string(index=False))

    con = []
    for b in base:
        for strat, suf in (("raw", ""), ("cal", "_cal"), ("anchor", "_anc")):
            for split in ("in_district", "held_out_district"):
                d = df[df["spatial_holdout"] == int(split == "held_out_district")]
                est, lo, hi = EV.paired_bootstrap(d, b + "_rk" + suf, b + suf)
                con.append(dict(model=b, strategy=strat, split=split, contrast="rank - raw counts",
                                d_brier=est, ci_lo=lo, ci_hi=hi))
    con = pd.DataFrame(con)
    con.to_csv(TABLES/"sec_rank_contrasts.csv", index=False)
    print(con.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for kind, strat, ls, lw in (("raw counts", "cal", "--", 1.3), ("rank", "cal", "-", 2.2),
                                ("raw counts", "anchor", ":", 1.3), ("rank", "anchor", "-.", 1.6)):
        d = byo[(byo["model"] == "M2e") & (byo["inputs"] == kind) & (byo["strategy"] == strat)]
        ax.plot(pd.to_datetime(d["origin"]), d["o_e"], ls, lw=lw, marker="o", ms=4,
                label="{}, {}".format(kind, "Platt 2023-01" if strat == "cal" else "anchored"))
    ax.axhline(1, color="#999", lw=1)
    ax.set_ylabel("Observed / expected (M2e, in-district)")
    ax.set_xticks(pd.to_datetime(origins("test")))
    ax.set_xticklabels([o[:7] for o in origins("test")])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES/"sec_rank_by_origin.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
