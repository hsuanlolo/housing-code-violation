# Pre-registration: prospective test of anchored, district-adjusted housing-hazard forecasts

**Project.** Before Signing the Lease: Predictive Value and Spatial Reliability of Housing Maintenance Records in New York City (Hsuan Lo).
**Document date.** 6 October 2026.
**Registration record.** `docs/prereg/PREREG_SPEC.json` stores the SHA-256 of this document, the SHA-256 of every analysis script, and the fixed model parameters. The git commit that adds it is the registration timestamp. A public deposit (OSF or AsPredicted) of this exact file is recommended for an independent timestamp; see §11.

---

## 1. Why a second pre-registration

The study's first pre-registration (`docs/FROZEN_SPEC.json`, commit `8db7aa6`) was tested once on 2024–2025. Several of its results then prompted post-freeze analyses on those same years:

* ACS tract context (M2e);
* rolling, partial-window and anchored recalibration;
* a sensitivity grid;
* rank-normalized inputs.

Those analyses found a configuration that looks markedly better calibrated: **M2d inputs, recalibrated on the last closed window, anchored to a completed recent event rate, with shrunken district offsets**. The 2024–2025 test years have now been used to choose it, so they cannot confirm it. This document fixes that configuration, its comparator and its tests **before** any new prediction origin is scored.

## 2. Full disclosure of what has been seen

Everything from origins 2020-01 to 2025-07 is **development data**. That covers features, outcomes, every model in paper Sections 7.2–7.9, and a dry run of this exact procedure on origin 2025-07 (§10). In particular:

| Analysis | Data used | Status |
|---|---|---|
| Frozen test (H1–H4b, warning rules) | 2024-01 … 2025-07 | confirmatory, done |
| ACS context, recalibration, anchoring, sensitivity grid, rank inputs | 2023-01 … 2025-07 | exploratory |
| Dry run of this procedure | origin 2025-07 | code test |

**Data-access statement for the registered origins.**

* The data download used so far (6 Oct 2026; violations through 4 Oct 2026) physically contains the complete outcome window of origin **2026-01** and about three months of origin **2026-07**. No outcome, event rate or model score has been computed for either origin.
* Two artifacts touched these periods. Figure 1 shows a total 2026 Class C count (200,086, a calendar-year aggregate). Features at origin 2026-07 were computed for the RentCast coverage analysis. Features are not outcomes.
* Origins **2027-01** and **2027-07** are fully prospective. Their outcome windows have not begun.

## 3. Hypotheses

**Notation.**

* *Product*: M2d with the full procedure in §5.
* *Comparator*: M2c with Platt recalibration on the last closed window, which is the deployable version of the frozen study's preferred model.
* O/E: observed / mean predicted.
* District error: mean over community districts of |O/E − 1|.
* Rows: *in-district* rows are buildings outside the 12 held-out districts; *all* includes both.

### Primary (confirmatory, Holm-corrected across P1–P3, family-wise α = 0.05)

| | Hypothesis | Statistic (product − comparator) | Rows | Supported if |
|---|---|---|---|---|
| **P1** | Anchoring gives a citywide level closer to the truth | \|O/E − 1\| | in-district | one-sided bootstrap p below its Holm threshold |
| **P2** | District offsets give more even calibration across districts | district error | all | as above |
| **P3** | The product forecasts better overall | Brier score | in-district | as above |

### Secondary (each at one-sided α = 0.05, uncorrected, reported as secondary)

| | Hypothesis | Statistic | Rows |
|---|---|---|---|
| **S1** | H1 replicates: event history beats counts | Brier(M2b) − Brier(M2a) < 0 | in-district |
| **S2** | H2 replicates: complaints add information | Brier(M2c) − Brier(M2b) < 0 | in-district |
| **S3** | Rank-normalized district inputs calibrate districts better without anchoring | district error(M2d_rk) − district error(M2d), both Platt on last closed window | all |
| **S4** | The district-relative warning rule (top 10% within each district, on the product score) leaves no district below 10% recall, at a citywide recall cost ≤ 20% vs the citywide rule | count and cost | all |

### Exploratory (reported, no hypothesis)

* Every model's Brier and O/E for held-out districts.
* M2e (ACS context) vs product.
* Completion-factor accuracy, i.e. completed rate vs realized rate at the anchoring origin.

## 4. Data and outcome

As in the frozen study, unchanged:

* **Outcome.** At least one HPD Class C violation inspected in [t, t + 6 months).
* **Features.** Records approved ≤ t − 30 days; complaint months ended ≤ t − 30 days.
* **Eligibility.** Any Class C or complaint in the 24 months before the cutoff.
* **Sources and geography.** HPD open data (`wvxf-dwi5`, `ygpa-z7cr`, `kj4p-ruqc`); community districts from the HPD register; ACS 2014–2018 tract context.

New data are downloaded with `make data` (code/01_fetch_hpd.py) at the times in §8.

## 5. Procedure for an origin t (fixed; nothing is tuned)

| Step | Specification |
|---|---|
| Training origins | Every half-year origin from 2020-01 to t − 18 months, in-district rows only |
| Models | M2a, M2b, M2c, M2d, M2e, M2d_rk with the feature sets of `04_models.py` / `10_acs_context.py` / `14_rank_features.py`. Penalty C fixed at the values in `PREREG_SPEC.json`, as selected in development |
| Recalibration | Platt (logistic on the logit score) fitted on origin L = t − 12 months, in-district rows, full outcomes (window closed by t − 6 months) |
| Completion factor | cf = event rate at L ÷ partial event rate at L. The partial rate counts only events inspected and approved by L + 6 months − 30 days |
| Anchoring target | r* = cf × partial event rate at L′ = t − 6 months (events inspected and approved by t − 30 days), in-district rows |
| Anchor | Intercept shift so that the mean M2d prediction over origin t's **own** in-district rows equals r*. Uses t's features only |
| District offsets | Per community district, a penalized (λ = 100) logit shift moving L′ predictions toward cf × that district's partial count. Applied to t's predictions |
| Warning rule | Top 10% of product scores within each community district |
| Held-out districts | The same 12 districts as the frozen study. Never in training or recalibration rows. Their own past partial counts set their district offsets |

Implementation: `code/15_prereg.py predict <t>`. It sets the outcome columns of origin t to missing before anything is saved, and writes the predictions plus their SHA-256.

## 6. Analysis plan

* **Primary metrics:** as in §3. Evaluation: `code/15_prereg.py evaluate <t>`.
* **Inference:** bootstrap over buildings (one row per building per origin), B = 1,000, seed 20261006. One-sided p = (#{bootstrap statistic ≥ 0} + 1)/(B + 1).
* **Multiplicity:** Holm across P1–P3 within each origin.
* **Decision.** Each Tier A origin is tested separately.
  * **Confirmed:** a primary hypothesis is supported at 2027-01 **and** its estimate has the same sign at 2027-07.
  * **Not confirmed:** it fails at 2027-01, whatever 2027-07 shows.
  * **Weight of Tier B:** Tier B results are reported in full but carry less weight (§2).
* **Data problems.** If HPD changes the dataset schema or the Class C definition between now and evaluation, we will report it, adapt only the fetch step, and run the stable-category outcome (paper §7.8) as an added sensitivity analysis. The primary definition will not change.

## 7. Guards (enforced in code)

1. `register` writes the hashes of this document, the code and the parameters. `predict` and `evaluate` refuse to run if any of them has changed.
2. `predict` writes `docs/prereg/pred_<t>.sha256`, which is committed before evaluation. `evaluate` refuses if the predictions file does not match.
3. `evaluate` refuses unless the violations data were downloaded on or after t + 6 months + 60 days, so that late approvals are included.
4. Any deviation is recorded in `docs/decisions.md` and §12 below **before** evaluation, and re-registered under a new hash. The original registration stays in the git history.

## 8. Timeline

| Origin | Tier | Predict (download data) | Commit prediction hash by | Evaluate with data downloaded on or after |
|---|---|---|---|---|
| 2026-01-01 | B, sealed backtest | immediately after registration (current download) | before evaluating | now (current download 2026-10-06 ≥ 2026-08-30) |
| 2026-07-01 | B, sealed backtest | immediately after registration (current download) | before evaluating | 2027-03-02 |
| **2027-01-01** | **A, primary** | download between 2026-12-02 and 2026-12-31 | **2026-12-31** | **2027-08-30** |
| 2027-07-01 | A, replication | download between 2027-06-01 and 2027-06-30 | 2027-06-30 | 2028-02-29 |

For context, the AAG abstract can be edited until 17 Dec 2026. Only Tier B (2026-01) can be evaluated before then.

## 9. What would count against the approach

* **P1 fails:** anchoring to a completed recent rate does not fix the citywide level prospectively. The anchor target's own error, which was ±0.01 in development and over-shot by 0.019 at 2025-07, dominates.
* **P2 fails:** district offsets estimated from partial windows do not transfer, and the district-level gains in Sections 7.7–7.9 were tuned to 2024–2025.
* **P3 fails:** calibration improves without improving accuracy. The product is then better described as a presentation choice than a better forecast.
* **S1–S2 fail:** the core claim of the paper weakens out of sample.

## 10. Development dry run (origin 2025-07, development data)

The full procedure was run on origin 2025-07 with `--dev` (outputs in `out/prereg_dev/`). This is a code test on data already used to choose the procedure. It is **not** evidence for the hypotheses.

| | Product | Comparator (M2c, Platt on L) |
|---|---|---|
| O/E, in-district | 0.925 | 0.921 |
| Brier, in-district | 0.1075 | 0.1084 |
| Brier, held-out districts | 0.1011 | 0.1016 |
| District error (all) | 0.097 | 0.149 |

* **Primary contrasts:** P1 −0.004, P2 −0.051 and P3 −0.0008, all one-sided p ≈ 0.001.
* **Secondary contrasts:** S1 −0.011 and S2 −0.006, both supported. S3 +0.0001, not supported. S4: districts below 10% recall go from 10 to 0, at a 15.8% recall cost.
* **What the run shows:** the anchoring target over-shot (0.244 vs 0.225), so the level gain was small. P1 is the hypothesis most likely to fail prospectively.

## 11. Public timestamp (author action)

Git commit times are self-reported. Before any registered origin is scored, the author should deposit this file and `PREREG_SPEC.json` on OSF (or AsPredicted). The resulting URL and DOI are recorded in `docs/prereg/public_deposit.md`. That file is deliberately not hashed, so recording the deposit does not alter this registration.

## 12. Deviations

*None at registration.*
