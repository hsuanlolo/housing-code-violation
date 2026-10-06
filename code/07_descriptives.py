"""Descriptive exhibits: Figure 1 (Class C by year) and Table 4 (panel by origin).

Figure 1 queries yearly counts from the API directly (back to 2012, beyond the
2017+ download) so the enforcement-drift context is not limited by the panel.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

from common import DS_VIOLATIONS, FIGURES, FINAL, SOCRATA, TABLES


def fig1():
    r = requests.get("{}/{}.json".format(SOCRATA, DS_VIOLATIONS), params={
        "$select": "date_extract_y(inspectiondate) AS y, class, count(*) AS n",
        "$where": "inspectiondate >= '2013-01-01' AND class in ('A','B','C')",
        "$group": "y, class", "$order": "y"}, timeout=300)
    r.raise_for_status()
    d = pd.DataFrame(r.json()).astype({"y": int, "n": int})
    d.pivot(index="y", columns="class", values="n").to_csv(TABLES/"violations_by_year_class.csv")
    c = d[d["class"] == "C"].set_index("y")["n"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(c.index, c.values / 1000, color=["#4a6fa5"] * (len(c) - 1) + ["#b8c4d6"], width=0.7)
    ax.set_ylabel("Class C violations (thousands)")
    ax.set_xticks(c.index)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(c.index[-1], c.values[-1] / 1000 + 5, "partial", ha="center", fontsize=7, color="#666")
    fig.tight_layout()
    fig.savefig(FIGURES/"fig1_classC_by_year.png", dpi=200)
    print(c.to_string())


def panel_table():
    p = pd.read_parquet(FINAL/"panel.parquet")
    t = p.groupby(["origin", "role"]).agg(
        buildings=("y", "size"), events=("y", "sum"), event_rate=("y", "mean"),
        any_c_last_year=("c365", lambda s: (s > 0).mean()),
        no_classC_history=("never_c", "mean")).reset_index()
    # Test-origin outcomes stay sealed until the freeze.
    t.loc[t["role"] == "test", ["events", "event_rate"]] = float("nan")
    t.to_csv(TABLES/"panel_by_origin.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    fig1()
    panel_table()
