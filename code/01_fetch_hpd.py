"""Download the three HPD datasets from NYC Open Data (no API key required).

  violations  Class C only, event level, 2017+  -> data/raw/violations_classC.csv
  complaints  aggregated server-side to building x month x category x type
                                                  -> data/raw/complaints_bym.csv
  buildings   full register                       -> data/raw/buildings.csv

A fetch log with row counts and the download timestamp is written to
data/raw/fetch_log.json. The open data portal is a moving target (records are
added, statuses change, some records may be withdrawn), so the timestamp is
part of the result.

Usage: python code/01_fetch_hpd.py [violations|complaints|buildings ...]
"""
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

from common import (DS_BUILDINGS, DS_COMPLAINTS, DS_VIOLATIONS, HISTORY_START,
                    RAW, SOCRATA)

PAGE = 50_000
WORKERS = 5

V_COLS = ["violationid", "buildingid", "bbl", "bin", "boroid", "communityboard",
          "nta", "censustract", "latitude", "longitude", "inspectiondate",
          "approveddate", "novissueddate", "ordernumber", "currentstatus",
          "rentimpairing"]


def get_csv(dataset, params, tries=5):
    url = "{}/{}.csv".format(SOCRATA, dataset)
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=300)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), dtype=str)
        except Exception as e:  # network hiccups and portal 5xx are common
            wait = 5 * 2 ** k
            print("  retry {} in {}s: {}".format(k + 1, wait, str(e)[:120]))
            time.sleep(wait)
    raise RuntimeError("giving up on {} {}".format(dataset, params))


def paged(dataset, base):
    out, off = [], 0
    while True:
        p = dict(base, **{"$limit": PAGE, "$offset": off})
        df = get_csv(dataset, p)
        out.append(df)
        off += len(df)
        print("  {} {:>9,} rows".format(base.get("$where", "")[-30:], off), flush=True)
        if len(df) < PAGE:
            break
    return pd.concat(out, ignore_index=True)


def _years():
    return range(int(HISTORY_START[:4]), datetime.now().year + 1)


def _year_where(col, y):
    return "{0} >= '{1}-01-01' AND {0} < '{2}-01-01'".format(col, y, y + 1)


def fetch_violations():
    # One paged query per inspection year, run in parallel: deep $offset pages
    # are slow on the portal, so many short scans beat one long one.
    def one(y):
        print(" violations", y, flush=True)
        return paged(DS_VIOLATIONS, {
            "$select": ",".join(V_COLS),
            "$where": "class='C' AND " + _year_where("inspectiondate", y),
            "$order": "violationid",
        })
    with ThreadPoolExecutor(WORKERS) as ex:
        df = pd.concat(list(ex.map(one, _years())), ignore_index=True)
    df = df.drop_duplicates("violationid")
    df.to_csv(RAW/"violations_classC.csv", index=False)
    return len(df)


def fetch_complaints():
    # Server-side GROUP BY, one calendar month per request: year-sized groups
    # time out on the portal, and every $offset page re-runs the aggregation.
    months = pd.date_range(HISTORY_START, datetime.now(), freq="MS")

    def one(m):
        nxt = m + pd.offsets.MonthBegin(1)
        df = paged(DS_COMPLAINTS, {
            "$select": ("building_id, date_trunc_ym(received_date) AS month, "
                        "major_category, type, count(*) AS n_problems, "
                        "count(distinct complaint_id) AS n_complaints"),
            "$where": "received_date >= '{:%Y-%m-%d}' AND received_date < '{:%Y-%m-%d}'"
                      .format(m, nxt),
            "$group": "building_id, month, major_category, type",
            "$order": "building_id, major_category, type",
        })
        return df
    with ThreadPoolExecutor(WORKERS) as ex:
        df = pd.concat(list(ex.map(one, months)), ignore_index=True)
    df.to_csv(RAW/"complaints_bym.csv", index=False)
    return len(df)


def fetch_violations_ab():
    """SECONDARY (post-freeze sensitivity): Class A/B counts by building x inspection month."""
    months = pd.date_range(HISTORY_START, datetime.now(), freq="MS")

    def one(m):
        nxt = m + pd.offsets.MonthBegin(1)
        return paged(DS_VIOLATIONS, {
            "$select": "buildingid, date_trunc_ym(inspectiondate) AS month, class, count(*) AS n",
            "$where": "class in ('A','B') AND inspectiondate >= '{:%Y-%m-%d}' AND inspectiondate < '{:%Y-%m-%d}'"
                      .format(m, nxt),
            "$group": "buildingid, month, class",
            "$order": "buildingid, class",
        })
    with ThreadPoolExecutor(WORKERS) as ex:
        df = pd.concat(list(ex.map(one, months)), ignore_index=True)
    df.to_csv(RAW/"violations_ab_bym.csv", index=False)
    return len(df)


def fetch_buildings():
    df = paged(DS_BUILDINGS, {"$order": "buildingid"})
    df.to_csv(RAW/"buildings.csv", index=False)
    return len(df)


JOBS = {"violations": fetch_violations, "complaints": fetch_complaints,
        "buildings": fetch_buildings}
OPTIONAL = {"violations_ab": fetch_violations_ab}   # run only when named

if __name__ == "__main__":
    todo = sys.argv[1:] or list(JOBS)
    JOBS.update(OPTIONAL)
    log_path = RAW/"fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for name in todo:
        print("fetching", name)
        n = JOBS[name]()
        log[name] = {"rows": n,
                     "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        log_path.write_text(json.dumps(log, indent=2))
        print("  done:", n, "rows")
