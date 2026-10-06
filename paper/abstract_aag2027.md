# AAG 2027 abstract

**Submission:** due **15 October 2026**. It can be edited until **17 December 2026**, so swap in test-set numbers after the freeze. Registration must be paid before submitting. Limit: 250 words, plain text (no bold, italics or bullets), no name or affiliation in the body, abbreviations spelled out.

**Suggested theme track:** Geographic Methods, Tools, and Techniques for Knowledge and Justice.

**Session type:** Paper. If it is still unassigned by 12 November it converts to a poster, so contact organizers of urban data, housing or GIScience sessions early.

---

**Title:** Before Signing the Lease: Predictive Value and Spatial Reliability of Housing Maintenance Records in New York City

**Abstract (246 words):**

Renters in New York City can look up a building's housing code violations before signing a lease, but a list of past violations does not say whether hazards will recur. This paper asks whether the timing, type, and repetition of public maintenance records forecast new immediately hazardous violations better than a simple count, and where in the city such forecasts can be trusted. Using every immediately hazardous violation and tenant complaint recorded by the city's housing agency since 2017, we build a panel of 832,000 building-half-years and compare a count baseline, nested logistic regressions, and gradient boosting. Forecasts use only records public thirty days before each forecast date, models are frozen before testing on 2024 and 2025, and one fifth of community districts is withheld from all fitting. In a development year, event history cut probability error by ten percent relative to counts, complaints cut it further, and gradient boosting added nothing measurable. Among the tenth of buildings ranked highest, eighty percent received a new hazardous violation within six months, against seventy-one percent when ranked by counts. Reliability was spatially uneven: the model overstated risk up to threefold in districts with few recorded hazards, and a single citywide warning threshold caught between six and sixty-three percent of eventual hazards depending on the district. Linking 3,885 rental listings shows that only three in five can receive any record-based warning. Public-data housing tools should report where they are reliable and treat missing records as missing evidence, not safety.

**Keywords:** housing quality, administrative data, predictive modeling, spatial reliability, New York City

---

### Notes for the 17 December edit
* Replace "In a development year" with the 2024–2025 test results. Keep the sentence structure so the word count holds.
* Each number above maps to `out/tables/dev_overall.csv`, `dev_contrasts.csv`, `dev_by_cd_M2c.csv` and `rentcast_coverage.csv`:
  * "cut probability error by ten percent": Brier M2b vs M2a, 0.1288 → 0.1162 (−9.8%).
  * "eighty percent … seventy-one percent": precision@10% M2c 0.799 vs M1r 0.715.
  * "up to threefold": district O/E minimum 0.33 (CD 503).
  * "six and sixty-three percent": within-district recall at the citywide top-10% cut, 0.062–0.625.
  * "three in five": 58.6% of listings in scope.
