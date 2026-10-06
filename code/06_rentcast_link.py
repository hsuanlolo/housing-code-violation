"""Link RentCast rental listings to HPD buildings and measure model coverage (RQ5).

Listings are matched to HPD building ids by normalized address (house number +
street + borough), falling back to the nearest HPD building point within
MAX_DIST_M. For each listing we then record whether the building is
  * not in HPD's records at all,
  * in HPD's records but out of model scope (no record in the last 24 months), or
  * in scope, with features as of APPLY_ORIGIN.

The RentCast file is private and is read in place, never copied into the repo.
Set RENTCAST_LISTINGS to its path (default: the ai-housing-audit project on the Desktop).

Risk scores are attached only after the model is frozen; until then this script
reports coverage, which is itself a result: it is the share of the rental market
for which any record-based warning can be offered at all.
"""
import importlib.util
import os
import pathlib
import re

import numpy as np
import pandas as pd

from common import FINAL, ROOT, TABLES

APPLY_ORIGIN = "2026-07-01"   # latest half-year origin before the listing snapshot
MAX_DIST_M = 30
DEFAULT = pathlib.Path.home()/"Desktop"/"Papers"/"AI fairness in real estate"/"data"/"final"/"listings.csv"
LISTINGS = os.environ.get("RENTCAST_LISTINGS", str(DEFAULT))

BORO = {"Manhattan": "1", "Bronx": "2", "Brooklyn": "3", "Queens": "4", "Staten Island": "5"}
SUFFIX = {"ST": "STREET", "AVE": "AVENUE", "AV": "AVENUE", "RD": "ROAD", "PL": "PLACE",
          "BLVD": "BOULEVARD", "PKWY": "PARKWAY", "DR": "DRIVE", "LN": "LANE", "CT": "COURT",
          "TER": "TERRACE", "HWY": "HIGHWAY", "SQ": "SQUARE", "TPKE": "TURNPIKE",
          "E": "EAST", "W": "WEST", "N": "NORTH", "S": "SOUTH"}


def norm_street(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())
    s = re.sub(r"\b(\d+)(ST|ND|RD|TH)\b", r"\1", s)          # 3RD -> 3
    return " ".join(SUFFIX.get(w, w) for w in s.split())


def norm_house(s):
    return re.sub(r"\s+", "", str(s).upper()).lstrip("0")


def parse_listing(addr):
    street_part = str(addr).split(",")[0]
    m = re.match(r"^\s*([\d\-]+[A-Z]?)\s+(.*)$", street_part.upper())
    return (norm_house(m.group(1)), norm_street(m.group(2))) if m else (None, None)


def load_panel_builder():
    spec = importlib.util.spec_from_file_location("bp", ROOT/"code"/"03_build_panel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    lst = pd.read_csv(LISTINGS)
    lst[["house", "street"]] = lst["formattedAddress"].apply(lambda a: pd.Series(parse_listing(a)))
    lst["boroid"] = lst["borough"].map(BORO)

    geo = pd.read_csv(FINAL/"buildings_geo.csv", dtype={"buildingid": str, "boroid": str})
    geo["house"] = geo["housenumber"].map(norm_house)
    geo["street"] = geo["streetname"].map(norm_street)
    key = geo.dropna(subset=["house", "street"]).drop_duplicates(["boroid", "house", "street"])
    m = lst.merge(key[["boroid", "house", "street", "buildingid"]],
                  on=["boroid", "house", "street"], how="left")
    m["match"] = np.where(m["buildingid"].notna(), "address", None)

    # Nearest-point fallback (equirectangular metres are fine at city scale).
    pts = geo.dropna(subset=["lat", "lon"])
    if len(pts):
        lat0 = np.deg2rad(40.7)
        px = np.deg2rad(pts["lon"].astype(float).values) * np.cos(lat0) * 6371000
        py = np.deg2rad(pts["lat"].astype(float).values) * 6371000
        todo = m["buildingid"].isna()
        for i in m.index[todo]:
            x = np.deg2rad(m.at[i, "longitude"]) * np.cos(lat0) * 6371000
            y = np.deg2rad(m.at[i, "latitude"]) * 6371000
            d = np.hypot(px - x, py - y)
            j = d.argmin()
            if d[j] <= MAX_DIST_M:
                m.at[i, "buildingid"] = pts["buildingid"].iat[j]
                m.at[i, "match"] = "nearest_{}m".format(MAX_DIST_M)

    bp = load_panel_builder()
    v, c = bp.load_violations(), bp.load_complaints()
    dev = c[c["month"] < "2023-07-01"]
    top = dev.groupby("major_category")["n_problems"].sum().nlargest(bp.TOP_COMPLAINT_CATS).index.tolist()
    feats = bp.build_origin(v, c, APPLY_ORIGIN, top)   # outcome columns are not yet observable
    feats = feats.drop(columns=["y", "y_count"])
    in_scope = set(feats["buildingid"])

    m["status"] = np.select(
        [m["buildingid"].isna(), m["buildingid"].isin(in_scope)],
        ["no_hpd_match", "in_scope"], default="hpd_match_out_of_scope")
    m = m.merge(feats, on="buildingid", how="left")
    m.to_parquet(FINAL/"listings_linked.parquet", index=False)

    cov = pd.crosstab(m["borough"], m["status"], margins=True, normalize="index").round(3)
    cov["n"] = pd.crosstab(m["borough"], m["status"], margins=True).sum(axis=1) // 2
    cov.to_csv(TABLES/"rentcast_coverage.csv")
    print(cov.to_string())
    print("\nmatch method:\n", m["match"].value_counts(dropna=False).to_string())
    m["rent_q"] = pd.qcut(m["rent"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    by_rent = pd.crosstab(m["rent_q"], m["status"], normalize="index").round(3)
    by_rent["c365>0"] = m.groupby("rent_q")["c365"].apply(lambda s: (s.fillna(0) > 0).mean()).round(3)
    by_rent.to_csv(TABLES/"rentcast_coverage_by_rent.csv")
    print(by_rent.to_string())


if __name__ == "__main__":
    main()
