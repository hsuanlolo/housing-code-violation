# Before Signing the Lease

**Predictive value and spatial reliability of housing-maintenance records in New York City.**

A renter can look up a building's HPD violation history. This project asks whether that history *forecasts* anything. Do the timing, type and repetition of public records predict a new **immediately hazardous (Class C)** violation in the next six months better than a simple count? And **where in the city** is such a forecast reliable?

Status: **v0.3 working paper. Spec frozen 6 Oct 2026 (commit `83b83e1`), 2024–2025 test scored once.** H1 (event history) and H2 (complaints) are supported; H3 (boosting) is supported but small (−0.7%); H4b (district context) did not replicate. Rankings transfer over time, but calibration decays as recorded violations per building inflate. Under a fixed 10% budget, the district-relative warning rule lifts every district to ≥19% recall at a 15% citywide recall cost.

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

## After the freeze: secondary analyses

Done (6 Oct 2026, labelled post-freeze; paper §7.7):
* **ACS tract context (M2e):** −0.9% Brier vs M2c (in- and out-of-district); districts > 20% miscalibrated fell from 20 to 9. Temporal drift unchanged.
* **Rolling recalibration:** half-yearly refresh lifts end-of-test O/E from 0.87 to 0.92. Per-district offsets from recent outcomes cut district error by about a third. Best: M2e + rolling district offsets.

* **Partial-window recalibration:** completed rates are accurate, but calibration did not improve. Diagnostic: the drift is **input inflation** (event rate flat, mean prediction rising 0.240 → 0.272).
* **Anchoring** (added after that diagnostic): match each origin's mean prediction to the completed recent rate. Pooled O/E 0.99. With district offsets, M2d reaches mean district error 0.028 and 0 districts > 20% off.
* **Sensitivity grid (8 variants):** H1/H2 hold everywhere; H3 small everywhere; H4b fails in-district everywhere. Calibration decay, the district gradient and the warning-rule trade-off all reproduce. The H1 gain halves when the outcome excludes the categories reclassified in 2019.

* **Rank-normalized inputs:** slow the decay but don't stop it, and cost a little Brier. Rank-normalized district context (M2d_rk) calibrates districts well with no recent outcomes.

## Pre-registered prospective test (registered 6 Oct 2026)

[`docs/preregistration.md`](docs/preregistration.md) and `docs/prereg/PREREG_SPEC.json`. The product (M2d + last-window Platt + anchoring + district offsets) is tested against M2c + last-window Platt.

| Origin | Tier | Predict | Commit hash by | Evaluate with data from |
|---|---|---|---|---|
| 2026-01 | B sealed backtest | now | before evaluating | now |
| 2026-07 | B sealed backtest | now | before evaluating | ≥ 2027-03-02 |
| **2027-01** | **A primary** | download 2026-12-02 … 12-31 | **2026-12-31** | ≥ 2027-08-30 |
| 2027-07 | A replication | download June 2027 | 2027-06-30 | ≥ 2028-02-29 |

```
python code/15_prereg.py predict 2027-01-01     # after `make data` in December 2026
git add docs/prereg/pred_2027-01-01.sha256 && git commit -m "Prereg predictions 2027-01"
python code/15_prereg.py evaluate 2027-01-01    # after `make data` on/after 2027-08-30
```

**Author action:** deposit `docs/preregistration.md` and `PREREG_SPEC.json` on OSF or AsPredicted before scoring any registered origin, and record the link in `docs/prereg/public_deposit.md`.

Next:
1. **Confirm the 2019 Class C reclassification** source and cite it (it now matters: about half the H1 gain rides on those categories).
2. **Run the registered tests** on the schedule above.
3. **RentCast:** hand-audit about 100 matches, then attach anchored scores and district-relative flags to listings.
4. **Figure:** community-district choropleth of O/E and of recall under both rules.
