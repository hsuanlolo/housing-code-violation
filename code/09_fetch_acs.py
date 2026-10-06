"""SECONDARY (post-freeze). Fetch ACS 2014-2018 5-year tract context and map buildings to tracts.

Why 2014-2018: it was released in December 2019, before the first prediction
origin (2020-01-01), so a single fixed vintage is known at every origin. It
uses 2010 tract geography. Being static, it cannot drift with enforcement,
which is the property this analysis tests (paper Section 7.7).

A Census API key is required. It is read from keys/census.txt in this repo, the
sibling ai-housing-audit project, or ~/.config/research, and is never printed:
every error message is scrubbed of it.

Outputs
  data/interim/acs2018_tract.csv    tract GEOID -> context features
  data/final/building_tract.csv     HPD buildingid -> tract GEOID
"""
import json
import pathlib
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from common import FINAL, INTERIM, RAW, ROOT

BASE = "https://api.census.gov/data/2018/acs/acs5"
COUNTY = {"1": "061", "2": "005", "3": "047", "4": "081", "5": "085"}   # HPD boroid -> county FIPS
KEY_PATHS = [ROOT/"keys"/"census.txt",
             pathlib.Path.home()/"Desktop"/"Papers"/"AI fairness in real estate"/"keys"/"census.txt",
             pathlib.Path.home()/".config"/"research"/"census.txt"]

VARS = {
    "B25035_001E": "median_year_built",
    "B25034_001E": "yb_total", "B25034_010E": "yb_1940s", "B25034_011E": "yb_pre1940",
    "B25003_001E": "tenure_total", "B25003_003E": "renter_occ",
    "B19013_001E": "median_hh_income",
    "B17001_001E": "pov_universe", "B17001_002E": "pov_below",
    "B25014_001E": "occ_total", "B25014_005E": "own_1to15", "B25014_006E": "own_15to2",
    "B25014_007E": "own_2plus", "B25014_011E": "rent_1to15", "B25014_012E": "rent_15to2",
    "B25014_013E": "rent_2plus",
    "B25002_001E": "hu_total", "B25002_003E": "hu_vacant",
    "B25024_001E": "uis_total", "B25024_008E": "uis_20to49", "B25024_009E": "uis_50plus",
}


def load_key():
    for p in KEY_PATHS:
        if p.is_file() and p.read_text().strip():
            return p.read_text().strip().strip('"').strip("'")
    raise SystemExit("No Census API key found (keys/census.txt).")


def fetch(key):
    out = []
    for fips in COUNTY.values():
        q = urllib.parse.urlencode({"get": ",".join(VARS), "for": "tract:*",
                                    "in": "state:36 county:{}".format(fips)})
        try:
            with urllib.request.urlopen("{}?{}&key={}".format(BASE, q, key), timeout=120) as r:
                body = r.read().decode()
        except Exception as e:
            raise SystemExit("ACS request failed: " + str(e).replace(key, "<REDACTED>"))
        if not body.startswith("["):   # 'Missing Key' / 'Invalid Key' come back as HTTP 200 HTML
            raise SystemExit("ACS returned a non-JSON body (key problem?)")
        rows = json.loads(body)
        out.append(pd.DataFrame(rows[1:], columns=rows[0]))
    df = pd.concat(out, ignore_index=True)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df = df.rename(columns=VARS)
    num = list(VARS.values())
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df[num] = df[num].where(df[num] >= 0)         # ACS uses large negatives for suppressed cells
    return df


def features(df):
    f = pd.DataFrame({"GEOID": df["GEOID"]})
    f["acs_median_year_built"] = df["median_year_built"]
    f["acs_share_pre1950"] = (df["yb_pre1940"] + df["yb_1940s"]) / df["yb_total"]
    f["acs_renter_share"] = df["renter_occ"] / df["tenure_total"]
    f["acs_log_income"] = np.log(df["median_hh_income"])
    f["acs_poverty_rate"] = df["pov_below"] / df["pov_universe"]
    crowd = df[["own_1to15", "own_15to2", "own_2plus", "rent_1to15", "rent_15to2", "rent_2plus"]].sum(axis=1)
    f["acs_crowding"] = crowd / df["occ_total"]
    f["acs_vacancy"] = df["hu_vacant"] / df["hu_total"]
    f["acs_share_20plus_units"] = (df["uis_20to49"] + df["uis_50plus"]) / df["uis_total"]
    return f.replace([np.inf, -np.inf], np.nan)


def tract_candidates(boro, code):
    """HPD writes tract 796.01 as '79601' and tract 104 as '104'. Return GEOID candidates."""
    if not isinstance(code, str) or not code.strip().isdigit() or boro not in COUNTY:
        return []
    c, pre = code.strip(), "36" + COUNTY[boro]
    cands = []
    if len(c) <= 4:
        cands.append(pre + c.zfill(4) + "00")      # integer tract
    if len(c) >= 3:
        cands.append(pre + c[:-2].zfill(4) + c[-2:])   # tract with two-digit suffix
    return cands


def map_buildings(valid):
    reg = pd.read_csv(RAW/"buildings.csv", dtype=str, usecols=["buildingid", "boroid", "censustract"])
    reg = reg[reg["buildingid"] != "0"]
    geoid, how = [], []
    for b, t in zip(reg["boroid"], reg["censustract"]):
        hit = [g for g in tract_candidates(b, t) if g in valid]
        geoid.append(hit[0] if hit else None)
        how.append("integer" if hit and hit[0].endswith("00") else ("suffix" if hit else "none"))
    reg["GEOID"], reg["tract_match"] = geoid, how
    return reg[["buildingid", "GEOID", "tract_match"]]


if __name__ == "__main__":
    raw = fetch(load_key())
    f = features(raw)
    f.to_csv(INTERIM/"acs2018_tract.csv", index=False)
    print("tracts:", len(f), " missing share by feature:")
    print(f.drop(columns="GEOID").isna().mean().round(3).to_string())
    bt = map_buildings(set(f["GEOID"]))
    bt.to_csv(FINAL/"building_tract.csv", index=False)
    print("\nbuilding -> tract match:", bt["tract_match"].value_counts(normalize=True).round(3).to_dict())
