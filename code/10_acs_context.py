"""SECONDARY (post-freeze). M2e: place context that does not come from enforcement records.

M2d's district features are averages of the same violation counts that
inflated in 2024-25, so their benefit decayed (paper Section 7.3). M2e replaces
them with fixed census-tract characteristics from ACS 2014-2018 (known before
the first origin, so they cannot drift with enforcement): housing age,
renter share, income, poverty, crowding, vacancy and large-building share.

  M2e = M2c + ACS tract context (+ missing flag)

Procedure mirrors the frozen test stage exactly: same held-out districts, C
tuned on train -> tune, fit on train+tune, Platt recalibration on 2023-01,
scored on the four test origins. This is not pre-specified and is labelled
as secondary everywhere it is reported.

Outputs: out/predictions_sec_acs.parquet, out/tables/sec_acs_*.csv,
         out/model_meta_sec_acs.json
"""
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import FINAL, INTERIM, OUT, TABLES, origins

ACS = ["acs_median_year_built", "acs_share_pre1950", "acs_renter_share", "acs_log_income",
       "acs_poverty_rate", "acs_crowding", "acs_vacancy", "acs_share_20plus_units"]
RAW_SCALE = ("trend", "comp_trend", "half_jul", "never_c")


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), "code/" + name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M, EV = load("04_models.py"), load("05_evaluate.py")


def make_lr(C, cols):
    """As the frozen make_lr, with ACS and district features left on their own scale."""
    def tf(X):
        return np.column_stack([
            X[:, i] if (c in RAW_SCALE or c.startswith(("cd_", "acs_"))) else np.log1p(np.clip(X[:, i], 0, None))
            for i, c in enumerate(cols)])
    return make_pipeline(FunctionTransformer(tf), StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def attach_acs(panel, train_mask):
    bt = pd.read_csv(FINAL/"building_tract.csv", dtype=str)[["buildingid", "GEOID"]]
    acs = pd.read_csv(INTERIM/"acs2018_tract.csv", dtype={"GEOID": str})
    panel = panel.merge(bt, on="buildingid", how="left").merge(acs, on="GEOID", how="left")
    panel["acs_missing"] = panel[ACS].isna().any(axis=1).astype(int)
    boro = (panel["cd"] // 100).fillna(0)
    # Impute with borough medians computed on training rows only.
    med = panel[train_mask].assign(boro=boro[train_mask]).groupby("boro")[ACS].median()
    city = panel.loc[train_mask, ACS].median()
    for c in ACS:
        fill = boro.map(med[c]).fillna(city[c])
        panel[c] = panel[c].fillna(fill)
    return panel


def main():
    panel = pd.read_parquet(FINAL/"panel.parquet")
    held = M.spatial_holdout(panel)
    panel["spatial_holdout"] = panel["cd"].isin(held).astype(int)
    fit_roles = origins("train") + origins("tune")
    train_mask = (panel["spatial_holdout"] == 0) & panel["origin"].isin(origins("train"))
    panel = attach_acs(panel, train_mask)

    cols = M.feature_sets(panel.columns)["M2c"] + ACS + ["acs_missing"]
    inside = panel[panel["spatial_holdout"] == 0]
    train = inside[inside["origin"].isin(origins("train"))]
    tune = inside[inside["origin"].isin(origins("tune"))]
    calib = inside[inside["origin"].isin(origins("calib"))]
    fit = inside[inside["origin"].isin(fit_roles)]

    best = None
    for C in M.LR_GRID:
        m = make_lr(C, cols).fit(train[cols].values, train["y"].values)
        b = brier_score_loss(tune["y"], m.predict_proba(tune[cols].values)[:, 1])
        if best is None or b < best[0]:
            best = (b, C)
    model = make_lr(best[1], cols).fit(fit[cols].values, fit["y"].values)
    f, ab = M.recalibrate(model.predict_proba(calib[cols].values)[:, 1], calib["y"].values)

    test = panel[panel["origin"].isin(origins("test"))].copy()
    test["M2e"] = model.predict_proba(test[cols].values)[:, 1]
    test["M2e_cal"] = f(test["M2e"].values)

    frozen = pd.read_parquet(OUT/"predictions_test.parquet")
    df = frozen.merge(test[["buildingid", "origin", "M2e", "M2e_cal", "acs_missing"]],
                      on=["buildingid", "origin"], how="left")
    assert df["M2e"].notna().all()
    df.to_parquet(OUT/"predictions_sec_acs.parquet", index=False)
    coef = dict(zip(cols, map(float, model.steps[-1][1].coef_[0])))
    (OUT/"model_meta_sec_acs.json").write_text(json.dumps(
        {"C": best[1], "recal": ab, "acs_coef": {k: v for k, v in coef.items() if k.startswith("acs_")},
         "acs_missing_share_test": float(df["acs_missing"].mean())}, indent=2))

    meta = json.loads((OUT/"model_meta_test.json").read_text())
    base = meta["base_rate_train"]
    rows = []
    for split, d in [("in_district", df[df["spatial_holdout"] == 0]),
                     ("held_out_district", df[df["spatial_holdout"] == 1])]:
        for c in ["M2c", "M2d", "M2e", "M2c_cal", "M2d_cal", "M2e_cal"]:
            rows.append(dict(split=split, model=c, **EV.metrics(d, c, base)))
    ov = pd.DataFrame(rows)
    ov.to_csv(TABLES/"sec_acs_overall.csv", index=False)
    print(ov[["split", "model", "brier", "brier_skill", "precision_at_10", "recall_at_10", "auc", "o_e_ratio"]]
          .round(4).to_string(index=False))

    con = []
    for a, b in [("M2e", "M2c"), ("M2e", "M2d"), ("M2e_cal", "M2c_cal")]:
        for split in ("in_district", "held_out_district"):
            d = df[df["spatial_holdout"] == int(split == "held_out_district")]
            est, lo, hi = EV.paired_bootstrap(d, a, b)
            con.append(dict(contrast="{} - {}".format(a, b), split=split, d_brier=est, ci_lo=lo, ci_hi=hi))
    con = pd.DataFrame(con)
    con.to_csv(TABLES/"sec_acs_contrasts.csv", index=False)
    print(con.to_string(index=False))

    dist = []
    for c in ["M2c", "M2d", "M2e", "M2c_cal", "M2d_cal", "M2e_cal"]:
        by = EV.by_district(df, c)
        e = (by["o_e_ratio"] - 1).abs()
        dist.append(dict(model=c, oe_median=by["o_e_ratio"].median(), oe_min=by["o_e_ratio"].min(),
                         oe_max=by["o_e_ratio"].max(), mean_abs_err=e.mean(),
                         mean_abs_err_heldout=e[by["held_out"] == 1].mean(), n_gt_20pct=int((e > 0.2).sum()),
                         rho_oe_rate=by[["o_e_ratio", "event_rate"]].corr("spearman").iloc[0, 1]))
    dist = pd.DataFrame(dist)
    dist.to_csv(TABLES/"sec_acs_district.csv", index=False)
    print(dist.round(3).to_string(index=False))

    ind = df[df["spatial_holdout"] == 0]
    byo = ind.groupby("origin").apply(lambda g: pd.Series(
        {**{"oe_" + c: g["y"].mean() / g[c].mean() for c in ["M2c", "M2d", "M2e"]},
         **{"brier_" + c: np.mean((g[c] - g["y"]) ** 2) for c in ["M2c", "M2d", "M2e"]}})).reset_index()
    byo.to_csv(TABLES/"sec_acs_by_origin.csv", index=False)
    print(byo.round(4).to_string(index=False))
    print("ACS coefficients (standardized):", {k: round(v, 3) for k, v in coef.items() if k.startswith("acs_")})


if __name__ == "__main__":
    main()
