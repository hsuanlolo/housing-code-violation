# Research plan: Before Signing the Lease

*Consolidated from the planning conversation (ChatGPT, Sept–Oct 2026) and adapted to the data as actually retrieved on 6 Oct 2026. Where this file and the code disagree, the code (`code/common.py`) is authoritative and the difference should be logged in `decisions.md`.*

## 1. Question

A renter about to sign a lease can look up a building's HPD record. A raw count of past violations mixes old problems with current ones. Two buildings with ten violations each can differ completely: one had them years ago, the other keeps getting new heat and leak violations every month.

**Core question.** Among NYC buildings with recent HPD activity, do the *timing, type and repetition* of past records predict a new **Class C** (immediately hazardous) violation in the next six months better than a simple count does? **Where** in the city does that predictive ability hold, and where does it fail?

The outcome is a *recorded* violation. It is not the true condition of a unit. Recording depends on tenants complaining, inspectors gaining access and agency practice. That dependence is a subject of the study, not a nuisance to be assumed away.

## 2. Research questions and hypotheses

| RQ | Hypothesis | Test |
|---|---|---|
| RQ1. Does event history beat counts? | **H1.** Adding recency, repetition and problem type improves prediction over counts. | Brier(M2b) < Brier(M2a) on the same rows; building-clustered paired bootstrap. |
| RQ2. Do complaints add information? | **H2.** Conditional on violations, complaint history still improves prediction. | Brier(M2c) < Brier(M2b). |
| RQ3. Is a complex model worth it? | **H3.** Gradient boosting beats logistic regression on identical inputs. | Brier(M3) < Brier(M2c), out-of-time *and* out-of-district. Secondary hypothesis; may fail. |
| RQ4. Is reliability spatially uneven? | **H4.** Performance falls in held-out districts, and calibration error is larger where records are sparser. **H4b.** District context known at *t* reduces district-level miscalibration. | Held-out vs in-district metrics at the same origin; per-district observed/expected ratio against record density; M2c vs M2d. |
| RQ5 (application). What can a listing platform add? | Descriptive. | Share of RentCast listings that match an HPD building and fall inside model scope, by borough and rent quintile. |

H1 and H2 are primary. H3 and H4 are secondary. RQ5 is the bridge to the product and is descriptive only.

## 3. Unit, timing and outcome

* **Unit:** building × prediction origin *t*. The origins are 1 January and 1 July of each year.
* **Outcome:** $Y_{it} = 1$ if building *i* receives at least one Class C violation with an inspection date in $[t, t+6\text{ months})$.
* **Information set at *t*:** violations approved by $t - 30$ days, and complaint months that ended by $t - 30$ days. The 30-day buffer absorbs publication lag. Sensitivity runs will use 0 and 60 days.
* **Eligibility:** any Class C violation or HPD complaint in the 24 months before the cutoff. Buildings outside this set are labelled *out of scope*, never *low risk*.

Example. At *t* = 1 Jan 2024 we use records approved by 2 Dec 2023 to predict whether the building gets a Class C violation between 1 Jan and 30 Jun 2024. A violation inspected on 15 Mar 2024 counts toward the outcome only, never toward the features.

## 4. Data (all public, verified 6 Oct 2026)

| Dataset | NYC Open Data id | Use | Verified |
|---|---|---|---|
| Housing Maintenance Code Violations | `wvxf-dwi5` | Features and outcome (Class C, 2017+) | 2.61M Class C rows. Yearly Class C counts run from 91k (2015) to 298k (2024). Has `buildingid`, BBL, BIN, community board, NTA, coordinates, inspection and approval dates. |
| Housing Maintenance Code Complaints and Problems | `ygpa-z7cr` | Complaint features | 2017–2026: 0.47–0.86M problems/yr and 0.30–0.53M complaints/yr. Has `building_id`, so it joins to violations without geocoding. |
| Buildings Subject to HPD Jurisdiction | `kj4p-ruqc` | Building register: community district, BBL, address | 380k rows |
| RentCast listings (private; from the `ai-housing-audit` project) | — | RQ5 application | 3,885 cleaned NYC listings (Aug 2026 snapshot), with address, coordinates, rent and bedrooms |

**Not used, on purpose:** `currentstatus` (today's status, so it would leak), and register fields such as units, stories and management program (current values, not as of *t*). Both can enter sensitivity analyses that are labelled as such.

**Known data-generating issues to report, not hide:**
1. **Enforcement drift.** Class C counts roughly tripled between 2015 and 2024. Base rates in the test years will differ from training, which is why calibration is a primary outcome.
2. **Dismissed violations.** Around 24% of 2019+ Class C records are currently "VIOLATION DISMISSED". They are included in the main analysis, because whether a violation would later be dismissed was unknown at *t*. A sensitivity run drops them.
3. **Back-filling.** The portal shows today's version of history. Records withdrawn before our download cannot be recovered.

## 5. Methods: three layers, compared and not chained

Fit order is M1 → M2 → M3. Each later layer must *earn* its complexity against the earlier one, on the same rows.

### M1: historical-count baseline
Bin past-year Class C count into {0, 1, 2–4, 5+} × season (Jan/Jul origin). The predicted probability is the bin's smoothed training event rate:

$$\hat p^{(1)}_{it} = \frac{E_g + \kappa\,\bar p}{N_g + \kappa}, \qquad \kappa = 20$$

Example: a bin has 100 observations and 20 events, and the overall rate is 10%. Then $\hat p = (20 + 2)/(100 + 20) = 18.3\%$. A ranking-only variant (M1r) sorts by raw count. This is what a renter does by eye.

### M2: logistic regression, three nested specifications
$$\hat p^{(2)}_{it} = \sigma(\eta_{it}), \quad \eta_{it} = \alpha + \beta_1 \log(1+C^{365}) + \beta_2 \log(1+C^{90}) + \beta_3 \log(1+D) + \theta^\top Z + \gamma\,\text{Jul}$$

* **M2a:** counts and season.
* **M2b:** adds days since the last Class C (*D*), the 2-year count, distinct inspection days and months (burst versus repeated), rent-impairing count, problem-type counts (11 types mapped from HMC order numbers), history length, and the trend term.
* **M2c:** adds complaint counts (90/365/730 days), emergency complaints, complaint months, complaint trend and complaint categories.

Trend is a per-day rate difference, not a raw subtraction:
$$\text{trend} = \frac{C^{90}}{90} - \frac{C^{365} - C^{90}}{275}$$
Example: 9 events in the last 90 days and 9 in the 275 days before. The last 90 days are about 3× the earlier daily rate, and trend ≈ +0.067/day.

The L2 penalty is tuned on the tune origin. Coefficients describe conditional associations, not causal effects.

### M2d: district context (added 6 Oct 2026, before the freeze)
M2c plus three features describing the building's community district at *t*: the share of in-scope buildings with any Class C in the last 365 and 90 days, and the mean log past-year complaints. They are computed from features, never outcomes. Held-out districts get their own values. M2d is motivated by the dev-stage finding that a place-blind model's district O/E tracks the district event rate (ρ = 0.80). Contrast: M2d − M2c, in-district and in held-out districts (H4b).

### M3: gradient boosting
$$F_M(X) = F_0 + \nu \sum_{m=1}^{M} f_m(X), \qquad \hat p^{(3)} = \sigma(F_M(X))$$
M3 uses the M2c inputs, with early stopping and a small grid over learning rate and leaf count. If it does not beat M2c robustly, M2c is the model we would ship.

## 6. Validation

| Role | Origins |
|---|---|
| Train | 2020-01 … 2022-01 (5) |
| Tune | 2022-07 |
| Calibrate | 2023-01 |
| (unused buffer) | 2023-07 |
| **Test (sealed)** | 2024-01, 2024-07, 2025-01, 2025-07 |

* **Temporal:** the model is frozen at end-2023 and scored once on 2024–25. `04_models.py test` refuses to run until `08_freeze.py` has written a spec hash.
* **Spatial:** 20% of community districts are withheld from every fitting stage. At each scoring origin we compare them with in-district buildings, which separates spatial transfer from temporal drift.
* **Metrics:** Brier score (primary) and Brier skill score against the training base rate. Precision and recall at a 10% review budget, AUC, calibration intercept and slope, ECE, and coverage.
* **Inference:** paired bootstrap that resamples buildings (B = 500).

Example for precision@10%: 1,000 buildings, 50 events. The model's top 100 contain 30 events and the count baseline's top 100 contain 20. Same inspection budget, ten more hazards found.

## 7. Expected outcomes and what each would mean

| Result | Research meaning | Product decision |
|---|---|---|
| History beats counts; boosting adds nothing | Feature construction matters more than model class | Ship logistic. It is cheap and explainable. |
| Boosting wins out-of-time and out-of-district | Non-linear structure transfers | Calibrated warning module |
| Good citywide, poor in some districts | One citywide score has a limited domain | Show per-area reliability; abstain where uncalibrated |
| Gains only at short horizons | Mostly predicting administrative continuation | Do not market it as a building-condition forecast |
| Nothing beats counts | Little incremental value in "AI" here | Keep the record lookup; do not sell a score |

## 8. Timeline (four weeks)

* **Week 1 (by Oct 12):** data, panel, M1, dev results. AAG abstract submitted by **Oct 15** (≤250 words, editable until Dec 17).
* **Week 2:** M2/M3, sensitivity runs (lag 0/60, dismissed excluded, Class B features), freeze.
* **Week 3:** test-set scoring once; spatial reliability analysis and map; RentCast linkage.
* **Week 4:** full draft; update the AAG abstract with test results before Dec 17.
