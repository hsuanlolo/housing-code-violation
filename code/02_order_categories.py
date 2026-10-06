"""Map HPD order numbers (Housing Maintenance Code sections) to problem types.

Class C violations carry ~126 distinct order numbers. Rather than download the
long free-text NOV description for 2M rows, we classify each order number once
from a representative description, using ordered keyword rules. The resulting
lookup (data/interim/order_categories.csv) is committed to the repo and is
meant to be reviewed by hand: the `rule` column says which keyword fired.

Input: data/raw/classC_ordernumbers.json (Socrata GROUP BY ordernumber, with
count and min(novdescription)); re-create it with --refresh.
"""
import json
import re
import sys

import pandas as pd
import requests

from common import DS_VIOLATIONS, HISTORY_START, INTERIM, RAW, SOCRATA

# First match wins, so specific rules precede general ones.
RULES = [
    ("lead_paint",   r"LEAD"),
    ("pests",        r"INFESTATION|ROACH|MICE|RODENT|\bRATS?\b|BEDBUG|BED BUG|VERMIN"),
    ("heat_water",   r"HEAT|HOT WATER|BOILER|SCALD|SUPPLY OF WATER|COLD WATER"),
    ("mold_leak",    r"MOLD|MOULD|LEAK|WATER DAMAGE|DAMP"),
    ("plumbing",     r"PIPE|DRAIN|WATER CLOSET|BATHTUB|BASIN|SINK|TOILET|PLUMBING|"
                     r"RADIATOR|OBSTRUCTION"),
    ("sanitation",   r"REFUSE|RUBBISH|CLEANSE|DISINFECT|STORAGE OF|COMBUSTIBLE|JANITOR"),
    ("detectors_guards", r"SMOKE|CARBON MONOXIDE|DETECTOR|WINDOW GUARD|SPRINKLER|FIRE"),
    ("doors_egress", r"SELF-CLOSING|SELF CLOSING|FASTENING|HASP|PADLOCK|EGRESS|"
                     r"FIRE ESCAPE|DOOR|LOCK|BARS? ON WINDOW|GATE"),
    ("gas_electric", r"\bGAS\b|ELECTRIC|WIRING|OUTLET"),
    ("structural",   r"STAIR|TREAD|PLASTER|CEILING|FLOOR|ROOF|WALL|STRUCTUR|COLLAPSE|"
                     r"RAILING|BANISTER|WINDOW"),
    ("admin_notice", r"POST NOTICE|CERTIFY|REGISTER|POST A SIGN|NAME AND LOCATION"),
]


def refresh():
    url = "{}/{}.json".format(SOCRATA, DS_VIOLATIONS)
    r = requests.get(url, params={
        "$select": "ordernumber, count(*) AS n, min(novdescription) AS d",
        "$where": "class='C' AND inspectiondate >= '{}'".format(HISTORY_START),
        "$group": "ordernumber", "$order": "n DESC", "$limit": 5000}, timeout=300)
    r.raise_for_status()
    (RAW/"classC_ordernumbers.json").write_text(r.text)


def classify(desc):
    d = (desc or "").upper()
    for cat, pat in RULES:
        m = re.search(pat, d)
        if m:
            return cat, m.group(0)
    return "other", ""


if __name__ == "__main__":
    if "--refresh" in sys.argv or not (RAW/"classC_ordernumbers.json").exists():
        refresh()
    rows = json.loads((RAW/"classC_ordernumbers.json").read_text())
    df = pd.DataFrame(rows).rename(columns={"d": "example_description"})
    df["n"] = df["n"].astype(int)
    df[["category", "rule"]] = df["example_description"].apply(
        lambda s: pd.Series(classify(s)))
    df["example_description"] = df["example_description"].str.slice(0, 160)
    df = df.sort_values("n", ascending=False)
    df[["ordernumber", "n", "category", "rule", "example_description"]].to_csv(
        INTERIM/"order_categories.csv", index=False)
    share = df.groupby("category")["n"].sum().sort_values(ascending=False)
    print((share / share.sum()).round(3).to_string())
