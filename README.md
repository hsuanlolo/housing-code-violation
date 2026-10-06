# Before Signing the Lease

**Predictive value and spatial reliability of housing-maintenance records in New York City.**

A renter can look up a building's HPD violation history. This project asks whether that history *forecasts* anything. Do the timing, type and repetition of public records predict a new **immediately hazardous (Class C)** violation in the next six months better than a simple count? And **where in the city** is such a forecast reliable?

Status: **v0.1 working paper, development stage.** The pipeline runs end to end on public data. The 2024–2025 test set is sealed until the model spec is frozen.

| | |
|---|---|
| Paper draft | [`paper/manuscript.md`](paper/manuscript.md) |
| AAG 2027 abstract (due **15 Oct 2026**, editable to 17 Dec) | [`paper/abstract_aag2027.md`](paper/abstract_aag2027.md) |
| Research plan (RQs, hypotheses, formulas, examples) | [`docs/research_plan.md`](docs/research_plan.md) |
| Design decisions log | [`docs/decisions.md`](docs/decisions.md) |

---

## Design in one screen

* **Unit:** building × prediction origin (1 Jan, 1 Jul).
* **Outcome:** ≥1 Class C violation inspected in the next 6 months.
* **Information set:** records approved ≥30 days before the origin. Never today's case status, never today's building attributes.
* **Scope:** buildings with any Class C or complaint in the prior 24 months. Everything else is "insufficient record", not "low risk".
* **Models, compared and not chained:** M1 count baseline → M2a/b/c/d nested logistic (counts → + event history and type → + complaints → + district context) → M3 gradient boosting on M2c inputs.
* **Validation:** train 2020–22, tune 2022-07, calibrate 2023-01, **test 2024–25 (sealed)**. 20% of community districts are withheld from all fitting.
* **Metrics:** Brier score (primary), calibration, precision and recall at a 10% review budget, AUC, coverage. Building-clustered paired bootstrap.

## Pipeline

```
make data     # 01 fetch HPD violations (Class C), complaints (aggregated), building register
              # 02 map 126 HMC order numbers -> 11 problem types
make panel    # 03 building x origin panel with leakage guards
make dev      # 04 fit M1-M3 on train, tune, score the dev origin; 05 evaluate
make link     # 06 link RentCast listings -> HPD buildings, coverage (RQ5)
make freeze   # 08 write docs/FROZEN_SPEC.json (commit it!)
make test     # 04/05 on sealed 2024-25 origins: refuses to run without the freeze
make paper    # 20 pandoc -> html/docx
```

Python 3.8+ (`requirements.txt`). Public data need no API key. RentCast listings are licensed and read in place from the sibling `ai-housing-audit` project, or from `$RENTCAST_LISTINGS`. They are never copied into this repo.

## Data

| Dataset | Id | Notes |
|---|---|---|
| HPD Housing Maintenance Code Violations | [`wvxf-dwi5`](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5) | Class C, 2017+ |
| HPD Complaints and Problems | [`ygpa-z7cr`](https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Complaints-and-Problems/ygpa-z7cr) | Aggregated server-side to building × month × category × type |
| Buildings Subject to HPD Jurisdiction | [`kj4p-ruqc`](https://data.cityofnewyork.us/Housing-Development/Buildings-Subject-to-HPD-Jurisdiction/kj4p-ruqc) | Community district, BBL, address |

`data/` is git-ignored except the order-number lookup. Re-running `make data` reproduces the data as of the new download date. The open-data portal is a moving target, so `data/raw/fetch_log.json` records when the data were pulled.

## Repo layout

```
code/     numbered pipeline scripts; common.py holds every design constant
docs/     research plan, decisions log, FROZEN_SPEC.json (after freeze)
paper/    manuscript.md, AAG abstract, build outputs (ignored)
out/      tables/ and figures/ (committed), predictions (ignored)
```

## Before the freeze: open decisions

These have to be settled and logged in `docs/decisions.md` **before** `make freeze`. After the freeze they can only be run as labelled sensitivity analyses.

1. ~~District context feature (M2d)~~ **Done (6 Oct).** It fixes systematic district miscalibration (ρ 0.80 → 0.10) but concentrates citywide warnings further. **New open decision:** which warning rule the paper recommends — a citywide threshold, district-relative thresholds, or probabilities shown alongside the district base rate. Decide before the freeze if it changes any primary metric.
2. **Confirm the 2019 Class C break.** Pests rose from ~0.7k to 46.7k, and doors/egress from 3.9k to 19.2k. Find the legal or administrative source and cite it. Decide whether the stable-category outcome is a sensitivity analysis (current plan) or the primary outcome.
3. **Sensitivity grid:** information lag 0/60 days; dismissed violations excluded; Class A/B counts as features; 3- and 12-month horizons.
4. **RentCast match audit:** hand-check about 100 listings, especially the 513 nearest-point (≤30 m) matches and Manhattan non-matches.
5. **Figure 3 map:** community-district choropleth of O/E (DCP boundary file).
