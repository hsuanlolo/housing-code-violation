# AAG 2027 abstract

**Submission:** due **15 October 2026**. It can be edited until **17 December 2026**. This version reports the frozen test results (2024–2025). Registration must be paid before submitting. Limit: 250 words, plain text (no bold, italics or bullets), no name or affiliation in the body, abbreviations spelled out.

**Suggested theme track:** Geographic Methods, Tools, and Techniques for Knowledge and Justice.

**Session type:** Paper. If it is still unassigned by 12 November it converts to a poster, so contact organizers of urban data, housing or GIScience sessions early.

---

**Title:** Before Signing the Lease: Predictive Value and Spatial Reliability of Housing Maintenance Records in New York City

**Abstract (246 words):**

Renters in New York City can look up a building's housing code violations before signing a lease, but a list of past violations does not say whether hazards will recur. This paper asks whether the timing, type, and repetition of public maintenance records forecast new immediately hazardous violations better than a simple count, and where in the city such forecasts can be trusted. Using every immediately hazardous violation and tenant complaint recorded by the city's housing agency since 2017, we compare a count baseline, nested logistic regressions, and gradient boosting on 832,000 building-half-years. Models use only records public before each forecast date, were frozen before testing on 2024 and 2025, and were evaluated in community districts withheld from fitting. Event history and complaints cut probability error by thirteen percent relative to counts; gradient boosting added under one percent. Among the tenth of buildings ranked highest, eighty-five percent received a new hazardous violation within six months, against seventy-eight percent ranked by counts. Rankings held, but calibration decayed: recorded violations per building rose forty percent while hazard rates stayed flat, inflating predicted risk. Forecasts overstated risk where few hazards are recorded, and a single citywide warning threshold flagged under a tenth of eventual hazards in eight districts. Spending the same warning budget within each district lifted every district above nineteen percent, at the cost of fifteen percent of hazards found citywide. Record-based housing tools should disclose their calibration over time, their reliability by place, and their warning rule.

**Keywords:** housing quality, administrative data, predictive modeling, spatial reliability, New York City

---

### Number sources (test origins, `out/tables/test_*.csv`)
* "thirteen percent": Brier M2a → M2c in-district, 0.1294 → 0.1121 (−13.4%).
* "under one percent": M3 − M2c, −0.7% in-district.
* "eighty-five … seventy-eight": precision@10% M2c 0.848 vs M1r 0.775 (in-district).
* "forty percent": mean past-year Class C per in-scope building, 2.32 (2023-01) → 3.22–3.37 (2024-07 to 2025-07); event rate 0.21–0.24.
* "eight districts": M2d, citywide top-10% rule, within-district recall < 0.10.
* "above nineteen percent … fifteen percent": district-relative rule, minimum district recall 0.194; overall recall 0.368 → 0.313 (−15%).
* Dropped from the earlier draft for length: RentCast coverage (58.6% of listings scorable).
