"""Build the building x origin panel: features known at t, outcome in (t, t+6m].

Leakage rules, enforced here and nowhere else:
  * A violation enters the features only if approveddate <= t - INFO_LAG_DAYS.
  * A complaint month enters the features only if the whole month ended
    on or before t - INFO_LAG_DAYS.
  * The outcome counts Class C violations with inspectiondate in [t, t+6m).
  * currentstatus is never used: it is today's status, not the status at t.
  * Building attributes from today's register (units, stories, program) are
    NOT features; they are used only to locate buildings for spatial analysis.

Eligibility at t: any Class C violation or any HPD complaint in the 24 months
before the information cutoff. Buildings with no recent record are out of
scope, not low risk.

Output: data/final/panel.parquet (one row per building x origin) and
data/final/buildings_geo.csv (building -> community district, NTA, tract, BBL).
"""
import numpy as np
import pandas as pd

from common import (FINAL, HORIZON_MONTHS, INFO_LAG_DAYS, INTERIM,
                    LOOKBACK_MONTHS, RAW, ROLE_OF, add_months, origins, ts)

TOP_COMPLAINT_CATS = 8


def load_violations():
    v = pd.read_csv(RAW/"violations_classC.csv", dtype=str)
    for c in ("inspectiondate", "approveddate"):
        v[c] = pd.to_datetime(v[c], errors="coerce")
    v = v.dropna(subset=["buildingid", "inspectiondate", "approveddate"])
    v = v[v["buildingid"] != "0"]
    cats = pd.read_csv(INTERIM/"order_categories.csv", dtype=str)
    v = v.merge(cats[["ordernumber", "category"]], on="ordernumber", how="left")
    v["category"] = v["category"].fillna("other")
    v["rentimpairing"] = (v["rentimpairing"] == "Y").astype(int)
    v["ym"] = v["inspectiondate"].dt.to_period("M")
    # Violations approved before they were inspected are data errors.
    v = v[v["approveddate"] >= v["inspectiondate"] - pd.Timedelta(days=1)]
    return v


def load_complaints():
    c = pd.read_csv(RAW/"complaints_bym.csv", dtype={"building_id": str})
    c = c.rename(columns={"building_id": "buildingid"})
    c = c[c["buildingid"].notna() & (c["buildingid"] != "0")]
    c["month"] = pd.to_datetime(c["month"])
    c["n_complaints"] = c["n_complaints"].astype(int)
    c["n_problems"] = c["n_problems"].astype(int)
    c["emergency"] = c["type"].fillna("").str.contains("EMERGENCY").astype(int)
    return c


def violation_features(v, t):
    cut = t - pd.Timedelta(days=INFO_LAG_DAYS)
    known = v[v["approveddate"] <= cut]
    g = known.groupby("buildingid")
    feat = pd.DataFrame({
        "last_c": g["inspectiondate"].max(),
        "first_c": g["inspectiondate"].min(),
    })
    for days in (90, 365, 730):
        w = known[known["inspectiondate"] >= t - pd.Timedelta(days=days)]
        feat["c{}".format(days)] = w.groupby("buildingid").size()
    w365 = known[known["inspectiondate"] >= t - pd.Timedelta(days=365)]
    gw = w365.groupby("buildingid")
    feat["c365_days"] = gw["inspectiondate"].nunique()
    feat["c365_months"] = gw["ym"].nunique()
    feat["c365_rentimp"] = gw["rentimpairing"].sum()
    cat = w365.pivot_table(index="buildingid", columns="category",
                           values="violationid", aggfunc="count")
    cat.columns = ["c365_cat_" + c for c in cat.columns]
    return feat.join(cat, how="left")


def complaint_features(c, t, top_cats):
    cut = t - pd.Timedelta(days=INFO_LAG_DAYS)
    known = c[c["month"] + pd.offsets.MonthBegin(1) <= cut]
    feat = pd.DataFrame(index=pd.Index(known["buildingid"].unique(), name="buildingid"))
    feat["first_comp"] = known.groupby("buildingid")["month"].min()
    for days in (90, 365, 730):
        w = known[known["month"] >= t - pd.Timedelta(days=days)]
        feat["comp{}".format(days)] = w.groupby("buildingid")["n_complaints"].sum()
    w365 = known[known["month"] >= t - pd.Timedelta(days=365)]
    feat["comp365_emerg"] = w365[w365["emergency"] == 1].groupby("buildingid")["n_problems"].sum()
    feat["comp365_months"] = w365.groupby("buildingid")["month"].nunique()
    mc = w365.assign(mc=np.where(w365["major_category"].isin(top_cats),
                                 w365["major_category"], "OTHER"))
    cat = mc.pivot_table(index="buildingid", columns="mc", values="n_problems", aggfunc="sum")
    cat.columns = ["comp365_cat_" + c.lower().replace("/", "_").replace(" ", "_")
                   for c in cat.columns]
    return feat.join(cat, how="left")


def outcome(v, t):
    end = add_months(t, HORIZON_MONTHS)
    w = v[(v["inspectiondate"] >= t) & (v["inspectiondate"] < end)]
    return w.groupby("buildingid").size().rename("y_count")


def build_origin(v, c, t, top_cats):
    t = ts(t)
    vf = violation_features(v, t)
    cf = complaint_features(c, t, top_cats)
    df = vf.join(cf, how="outer")
    lb = t - pd.Timedelta(days=INFO_LAG_DAYS) - pd.DateOffset(months=LOOKBACK_MONTHS)
    eligible = (df["last_c"] >= lb) | (df["comp730"].fillna(0) > 0)
    df = df[eligible].copy()
    df = df.join(outcome(v, t), how="left")
    df["y_count"] = df["y_count"].fillna(0).astype(int)
    df["y"] = (df["y_count"] > 0).astype(int)

    count_cols = [x for x in df.columns if x.startswith(("c90", "c365", "c730", "comp"))]
    df[count_cols] = df[count_cols].fillna(0)
    df["never_c"] = df["last_c"].isna().astype(int)
    df["days_since_c"] = (t - df["last_c"]).dt.days.fillna(3650).clip(upper=3650)
    first = pd.concat([df["first_c"], df["first_comp"]], axis=1).min(axis=1)
    df["history_months"] = ((t - first).dt.days / 30.44).fillna(0)
    # Rate difference in events/day: last 90 days vs the 275 days before.
    df["trend"] = df["c90"] / 90 - (df["c365"] - df["c90"]) / 275
    df["comp_trend"] = df["comp90"] / 90 - (df["comp365"] - df["comp90"]) / 275
    df["half_jul"] = int(t.month == 7)
    df["origin"] = t.strftime("%Y-%m-%d")
    df["role"] = ROLE_OF.get(df["origin"].iat[0], "apply")
    return df.drop(columns=["last_c", "first_c", "first_comp"]).reset_index()


def building_geo(v):
    """Locate each building in a community district for spatial analysis.

    Primary source: the HPD building register (boro + community board, block, lot).
    Fallback: the modal location recorded on its violations.
    """
    reg = pd.read_csv(RAW/"buildings.csv", dtype=str)
    reg = reg[reg["buildingid"] != "0"]
    reg["cd"] = pd.to_numeric(reg["boroid"], errors="coerce") * 100 + \
        pd.to_numeric(reg["communityboard"], errors="coerce")
    reg["bbl"] = (reg["boroid"].str.zfill(1) + reg["block"].str.zfill(5) +
                  reg["lot"].str.zfill(4))
    keep = ["buildingid", "cd", "bbl", "boroid", "housenumber", "streetname"]
    for col in ("zip", "bin"):
        if col in reg.columns:
            keep.append(col)
    geo = reg[keep].set_index("buildingid")

    vv = v.dropna(subset=["nta"])
    mode = vv.groupby("buildingid").agg(
        nta=("nta", lambda s: s.mode().iat[0]),
        censustract=("censustract", lambda s: s.mode().iat[0]),
        lat=("latitude", "first"), lon=("longitude", "first"),
        cd_v=("communityboard", lambda s: s.mode().iat[0]),
        boro_v=("boroid", "first"))
    mode["cd_v"] = pd.to_numeric(mode["boro_v"], errors="coerce") * 100 + \
        pd.to_numeric(mode["cd_v"], errors="coerce")
    geo = geo.join(mode.drop(columns="boro_v"), how="outer")
    geo["cd"] = geo["cd"].fillna(geo.pop("cd_v"))
    # Valid community districts are 1-18 within each borough.
    ok = geo["cd"].notna() & ((geo["cd"] % 100).between(1, 18))
    geo.loc[~ok, "cd"] = np.nan
    return geo.reset_index()


if __name__ == "__main__":
    v = load_violations()
    c = load_complaints()
    print("violations {:,}  complaint cells {:,}".format(len(v), len(c)))
    dev = c[c["month"] < "2023-07-01"]   # pick complaint categories on dev data only
    top_cats = dev.groupby("major_category")["n_problems"].sum() \
                  .nlargest(TOP_COMPLAINT_CATS).index.tolist()
    parts = []
    for t in origins():
        p = build_origin(v, c, t, top_cats)
        role = p["role"].iat[0]
        rate = "sealed" if role == "test" else "{:.3f}".format(p["y"].mean())
        print("  {}  {:<5}  n={:>7,}  base rate={}".format(t, role, len(p), rate))
        parts.append(p)
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.fillna({x: 0 for x in panel.columns if x.startswith(("c365_cat_", "comp365_cat_"))})
    geo = building_geo(v)
    geo.to_csv(FINAL/"buildings_geo.csv", index=False)
    panel = panel.merge(geo[["buildingid", "cd"]], on="buildingid", how="left")
    panel.to_parquet(FINAL/"panel.parquet", index=False)
    print("panel rows {:,}; buildings {:,}; missing CD {:.2%}".format(
        len(panel), panel["buildingid"].nunique(), panel["cd"].isna().mean()))
