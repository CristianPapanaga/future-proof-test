# Exploratory Data Analysis — Historical Concept Tests
*Generated 2026-10-07 12:09:24*

## 1. Dataset overview
The cleaned historical dataset contains one row per concept test. Currency columns were parsed to floats and categorical columns to category dtype (see ``data_utilities.py``).

| Key | Value |
| --- | --- |
| Concepts (rows) | 750 |
| Columns | 18 |
| Years covered | 2022–2025 |
| Concepts launched | 403 (53.7%) |
| Total research cost (€) | 8,633,060 |

## 2. Categorical distributions
Categorical variables are summarised as frequency tables (count and percentage of all concepts) rather than with means, since they are labels rather than measurements. ``Test_Year`` is shown chronologically rather than as a mean year.

### Product category
| Product category | Count | Pct (%) |
| --- | --- | --- |
| Snacks | 323 | 43.1 |
| Drinks | 269 | 35.9 |
| Personal care | 158 | 21.1 |

### Innovation type
| Innovation type | Count | Pct (%) |
| --- | --- | --- |
| Incremental improvement | 480 | 64 |
| New proposition | 270 | 36 |

### Research package
| Research package | Count | Pct (%) |
| --- | --- | --- |
| Survey | 290 | 38.7 |
| Behavioural | 274 | 36.5 |
| Combined | 186 | 24.8 |

### Research recommendation
| Research recommendation | Count | Pct (%) |
| --- | --- | --- |
| Progress | 269 | 35.9 |
| Stop | 245 | 32.7 |
| Refine | 236 | 31.5 |

### Test year
| Test year | Count | Pct (%) |
| --- | --- | --- |
| 2022 | 164 | 21.9 |
| 2023 | 214 | 28.5 |
| 2024 | 191 | 25.5 |
| 2025 | 181 | 24.1 |

### Launched
| Launched | Count | Pct (%) |
| --- | --- | --- |
| Yes | 403 | 53.7 |
| No | 347 | 46.3 |

### Research package by year
The research-package mix is tabulated against ``Test_Year`` to show how the design of concept tests has shifted over time.

| Test_Year | Survey | Behavioural | Combined |
| --- | --- | --- | --- |
| 2022 | 97 | 46 | 21 |
| 2023 | 104 | 79 | 31 |
| 2024 | 51 | 68 | 72 |
| 2025 | 38 | 81 | 62 |

| Test_Year | Survey | Behavioural | Combined |
| --- | --- | --- | --- |
| 2022 | 59.1 | 28.0 | 12.8 |
| 2023 | 48.6 | 36.9 | 14.5 |
| 2024 | 26.7 | 35.6 | 37.7 |
| 2025 | 21.0 | 44.8 | 34.3 |

| Research package | 2022 (%) | 2025 (%) | Change (pp) |
| --- | --- | --- | --- |
| Survey | 59.1 | 21.0 | -38.2 |
| Behavioural | 28.0 | 44.8 | 16.7 |
| Combined | 12.8 | 34.3 | 21.4 |

Between 2022 and 2025 the mix shifted decisively away from Survey-only research and toward the richer Behavioural and Combined packages. Survey fell from 59.1% to 21.0% (-38.2 pp), while Behavioural rose from 28.0% to 44.8% (+16.7 pp) and Combined rose from 12.8% to 34.3% (+21.4 pp).

A chi-square test of independence (year × package) gives χ² = 89.60, df = 6, p = 3.67e-17, so the change across years is statistically significant.

## 3. Numeric summary statistics
Each numeric column is summarised with its non-missing count, mean, standard deviation, minimum, quartiles (25/50/75) and maximum, plus the number of missing values. Mean and standard deviation summarise spread for roughly symmetric data, while the median and inter-quartile range are robust to the right-skew typical of cost variables. ``Research_Cost_EUR`` and ``Launch_Support_EUR`` are in euros.

| Column | Count | Mean | Std | Min | Q25 | Median | Q75 | Max | Missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sample_Size | 750 | 248.59 | 82.17 | 120 | 180 | 240 | 300 | 400 | 0 |
| Turnaround_Days | 750 | 15.29 | 3.73 | 7 | 13 | 15 | 18 | 26 | 0 |
| Research_Cost_EUR | 750 | 11510.75 | 2988 | 4480 | 9062.5 | 11335 | 13735 | 19730 | 0 |
| Stated_Appeal | 734 | 4.37 | 0.77 | 1.9 | 3.9 | 4.4 | 4.9 | 6.9 | 16 |
| Purchase_Intent | 730 | 3.9 | 0.87 | 1.2 | 3.3 | 3.9 | 4.5 | 7 | 20 |
| Behavioural_Choice_Pct | 431 | 29.44 | 11.98 | 0 | 21.6 | 30 | 37.5 | 65.4 | 319 |
| Implicit_Score | 167 | 51.3 | 13.91 | 16 | 41.95 | 51.3 | 61.45 | 86.4 | 583 |
| Launch_Support_EUR | 372 | 66070.7 | 23377.9 | 8000 | 50475 | 65050 | 81625 | 149500 | 378 |
| Distribution_Pct | 384 | 59.23 | 17.28 | 14.3 | 46.6 | 58.8 | 70.9 | 98 | 366 |
| Sales_vs_Target_Pct | 371 | 94.11 | 23.82 | 34.8 | 77.9 | 95.1 | 111 | 157 | 379 |
| Repeat_Purchase_Pct | 358 | 26.41 | 9.09 | 1.8 | 20.65 | 26.3 | 32.28 | 51.2 | 392 |

### Research cost and turnaround by package
``Research_Cost_EUR`` and ``Turnaround_Days`` are tabulated as means by research package, with Survey treated as the baseline. The difference columns show the premium (or saving) of the Behavioural and Combined packages relative to Survey. Means are reported here; the full distribution per column is available in the summary above.

| Research package | n | Mean cost (€) | Δ cost vs Survey (€) | Mean turnaround (days) | Δ turnaround vs Survey (days) |
| --- | --- | --- | --- | --- | --- |
| Survey | 290 | 8,767 | — | 12.1 | — |
| Behavioural | 274 | 11,982 | +3,214 | 15.8 | +3.8 |
| Combined | 186 | 15,094 | +6,326 | 19.5 | +7.5 |

### Correlations
Pearson correlations between all numeric columns are tabulated below (values rounded to 2dp). Each coefficient is computed on pairwise-complete observations, so pairs involving ``Implicit_Score`` (Combined packages only), ``Behavioural_Choice_Pct`` (Behavioural/Combined), or the launch metrics (launched concepts only) rest on smaller, more restricted subsamples and should be interpreted with that in mind.

The matrix is split into two tables — research-phase columns and launch-phase columns — so it fits the page width. Each table still lists all eleven variables as rows.

| Variable | Sample_Size | Turnaround_Days | Research_Cost_EUR | Stated_Appeal | Purchase_Intent | Behavioural_Choice_Pct | Implicit_Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sample_Size | 1 | 0.26 | 0.4 | -0.01 | -0.01 | -0.03 | -0.01 |
| Turnaround_Days | 0.26 | 1 | 0.81 | 0.03 | -0.01 | -0.03 | -0.02 |
| Research_Cost_EUR | 0.4 | 0.81 | 1 | 0.02 | -0.03 | 0.03 | 0.02 |
| Stated_Appeal | -0.01 | 0.03 | 0.02 | 1 | 0.49 | 0.38 | 0.3 |
| Purchase_Intent | -0.01 | -0.01 | -0.03 | 0.49 | 1 | 0.45 | 0.4 |
| Behavioural_Choice_Pct | -0.03 | -0.03 | 0.03 | 0.38 | 0.45 | 1 | 0.49 |
| Implicit_Score | -0.01 | -0.02 | 0.02 | 0.3 | 0.4 | 0.49 | 1 |
| Launch_Support_EUR | -0.09 | 0.37 | 0.37 | 0.06 | -0.04 | 0.11 | 0.05 |
| Distribution_Pct | -0.04 | 0.35 | 0.36 | -0.02 | 0.02 | 0.09 | -0.14 |
| Sales_vs_Target_Pct | -0.11 | 0.11 | 0.13 | 0.15 | 0.19 | 0.38 | 0.37 |
| Repeat_Purchase_Pct | 0 | 0.05 | 0.05 | 0.23 | 0.31 | 0.48 | 0.47 |

| Variable | Launch_Support_EUR | Distribution_Pct | Sales_vs_Target_Pct | Repeat_Purchase_Pct |
| --- | --- | --- | --- | --- |
| Sample_Size | -0.09 | -0.04 | -0.11 | 0 |
| Turnaround_Days | 0.37 | 0.35 | 0.11 | 0.05 |
| Research_Cost_EUR | 0.37 | 0.36 | 0.13 | 0.05 |
| Stated_Appeal | 0.06 | -0.02 | 0.15 | 0.23 |
| Purchase_Intent | -0.04 | 0.02 | 0.19 | 0.31 |
| Behavioural_Choice_Pct | 0.11 | 0.09 | 0.38 | 0.48 |
| Implicit_Score | 0.05 | -0.14 | 0.37 | 0.47 |
| Launch_Support_EUR | 1 | 0.31 | 0.26 | 0.14 |
| Distribution_Pct | 0.31 | 1 | 0.31 | 0.06 |
| Sales_vs_Target_Pct | 0.26 | 0.31 | 1 | 0.25 |
| Repeat_Purchase_Pct | 0.14 | 0.06 | 0.25 | 1 |

### Research-phase ↔ launch-phase correlations
The cross-phase correlations — between the pre-launch research metrics and the post-launch performance metrics — are shown below, restricted to pairs with |r| ≥ 0.20 and sorted by absolute value. Strength bands: |r| 0.20–0.29 weak, 0.30–0.49 moderate, ≥ 0.50 strong. These are the relationships most relevant to whether research metrics foreshadow launch outcomes.

| Research variable | Launch variable | r | Strength |
| --- | --- | --- | --- |
| Behavioural_Choice_Pct | Repeat_Purchase_Pct | 0.48 | Moderate |
| Implicit_Score | Repeat_Purchase_Pct | 0.47 | Moderate |
| Behavioural_Choice_Pct | Sales_vs_Target_Pct | 0.38 | Moderate |
| Turnaround_Days | Launch_Support_EUR | 0.37 | Moderate |
| Research_Cost_EUR | Launch_Support_EUR | 0.37 | Moderate |
| Implicit_Score | Sales_vs_Target_Pct | 0.37 | Moderate |
| Research_Cost_EUR | Distribution_Pct | 0.36 | Moderate |
| Turnaround_Days | Distribution_Pct | 0.35 | Moderate |
| Purchase_Intent | Repeat_Purchase_Pct | 0.31 | Moderate |
| Stated_Appeal | Repeat_Purchase_Pct | 0.23 | Weak |

- The behavioural and implicit measures are the strongest pre-launch predictors of launch outcomes: ``Behavioural_Choice_Pct`` and ``Implicit_Score`` correlate with ``Repeat_Purchase_Pct`` (r = 0.48 and 0.47) and ``Sales_vs_Target_Pct`` (r = 0.38 and 0.37), more strongly than any stated-metric pairing.
- ``Stated_Appeal`` now appears only once, and only weakly (r = 0.23 with ``Repeat_Purchase_Pct``), while ``Sample_Size`` is absent entirely: the attitudinal (stated) appeal and sample size still carry little predictive signal for launch outcomes.

### Notable correlations
The strongest pairwise correlations (by absolute value) are listed below, each classified as weak (0.20–0.29), moderate (0.30–0.49), or strong (≥ 0.50). Only one pair is strong: ``Turnaround_Days`` and ``Research_Cost_EUR`` (r = 0.81), reflecting that larger, more expensive studies take longer. The remainder are moderate (r ≈ 0.4–0.5): the research-phase metrics (``Stated_Appeal``, ``Purchase_Intent``, ``Behavioural_Choice_Pct``, ``Implicit_Score``) inter-correlate, and ``Repeat_Purchase_Pct`` tracks several of these, suggesting launch outcomes are partly foreshadowed by pre-launch consumer metrics.

| Variable 1 | Variable 2 | r | Strength |
| --- | --- | --- | --- |
| Turnaround_Days | Research_Cost_EUR | 0.81 | Strong |
| Stated_Appeal | Purchase_Intent | 0.49 | Moderate |
| Behavioural_Choice_Pct | Implicit_Score | 0.49 | Moderate |
| Behavioural_Choice_Pct | Repeat_Purchase_Pct | 0.48 | Moderate |
| Implicit_Score | Repeat_Purchase_Pct | 0.47 | Moderate |
| Purchase_Intent | Behavioural_Choice_Pct | 0.45 | Moderate |
| Sample_Size | Research_Cost_EUR | 0.4 | Moderate |
| Purchase_Intent | Implicit_Score | 0.4 | Moderate |
| Stated_Appeal | Behavioural_Choice_Pct | 0.38 | Moderate |
| Behavioural_Choice_Pct | Sales_vs_Target_Pct | 0.38 | Moderate |

## 4. Missing values
During cleaning, values that are legitimately absent are marked with the string ``"NA"``, while genuinely missing values remain ``pd.NA``. *Absent by design* rows are therefore those marked ``"NA"`` (e.g. launch-phase metrics for unlaunched concepts), and *applicable but missing* rows are the ``pd.NA`` values (a genuine data-collection gap). ``% missing`` is the collection-failure rate among applicable rows.

| Column | Applicable | Present | Absent by design | Applicable but missing | % missing |
| --- | --- | --- | --- | --- | --- |
| Concept_ID | 750 | 750 | 0 | 0 | 0 |
| Test_Year | 750 | 750 | 0 | 0 | 0 |
| Category | 750 | 750 | 0 | 0 | 0 |
| Innovation_Type | 750 | 750 | 0 | 0 | 0 |
| Research_Package | 750 | 750 | 0 | 0 | 0 |
| Sample_Size | 750 | 750 | 0 | 0 | 0 |
| Research_Cost_EUR | 750 | 750 | 0 | 0 | 0 |
| Turnaround_Days | 750 | 750 | 0 | 0 | 0 |
| Stated_Appeal | 750 | 734 | 0 | 16 | 2.1 |
| Purchase_Intent | 750 | 730 | 0 | 20 | 2.7 |
| Behavioural_Choice_Pct | 460 | 431 | 290 | 29 | 6.3 |
| Implicit_Score | 186 | 167 | 564 | 19 | 10.2 |
| Recommendation | 750 | 750 | 0 | 0 | 0 |
| Launched | 750 | 750 | 0 | 0 | 0 |
| Launch_Support_EUR | 403 | 372 | 347 | 31 | 7.7 |
| Distribution_Pct | 403 | 384 | 347 | 19 | 4.7 |
| Sales_vs_Target_Pct | 403 | 371 | 347 | 32 | 7.9 |
| Repeat_Purchase_Pct | 403 | 358 | 347 | 45 | 11.2 |

### Overall missingness
| Key | Value |
| --- | --- |
| Total cells | 13500 |
| Applicable cells | 11258 |
| Present cells | 11047 |
| Absent by design (valid, not missing) | 2242 |
| Truly missing cells (applicable but missing) | 211 |
| True missingness (% of all data) | 1.56% |

### Missingness mechanism
The overall true-missingness rate is low (1.56%), but it is far from uniform across fields. Two fields stand out: ``Implicit_Score`` (10.2% of applicable values missing) and ``Repeat_Purchase_Pct`` (11.2%), versus 2–8% elsewhere. This concentration is evidence against MCAR (missing completely at random), under which missingness would be spread evenly at a uniformly low rate with no relation to any variable.

Instead, the pattern is consistent with MAR or MNAR. ``Implicit_Score`` is only collected for Combined packages, and its elevated failure rate suggests the missingness depends on observed design features (MAR) or on the unobserved score itself (e.g. an implicit-association task that fails to yield a stable measure — MNAR). Likewise, ``Repeat_Purchase_Pct`` is measured only for launched concepts and appears to drop out for a non-random subset of them, plausibly tied to category, support spend, or the repeat-purchase behaviour itself.

Consequence: the concentration in ``Implicit_Score`` and ``Repeat_Purchase_Pct`` is worth flagging, but its practical impact hinges on the *overall* volume of missingness and on a formal test of the missingness mechanism — both addressed in section 5, which concludes that complete-case analysis is sufficient.

## 5. Little's MCAR test
Little's (1988) test evaluates the null hypothesis that missing values are Missing Completely At Random (MCAR); a small p-value (below 0.05) rejects MCAR in favour of MAR or MNAR. The test is run with ``pyampute.exploration.mcar_statistical_tests.MCARTest(method='little')``. Because the structural ``"NA"`` (absent by design) values are not statistically missing, each test is restricted to the subset of rows where its variables are actually applicable.

| Test | n | Variables | p-value | Conclusion |
| --- | --- | --- | --- | --- |
| Core research variables (all concepts) | 750 | Sample_Size, Turnaround_Days, Research_Cost_EUR, Stated_Appeal, Purchase_Intent | 0.333 | Fail to reject MCAR |
| Launch performance (launched concepts) | 403 | Launch_Support_EUR, Distribution_Pct, Sales_vs_Target_Pct, Repeat_Purchase_Pct | 0.303 | Fail to reject MCAR |
| Implicit/behavioural (Combined packages) | 186 | Behavioural_Choice_Pct, Implicit_Score, Stated_Appeal, Purchase_Intent | 0.694 | Fail to reject MCAR |

### Interpretation
**Core research variables (all concepts)** (n = 750) — p = 0.333: there is no evidence against MCAR.

**Launch performance (launched concepts)** (n = 403) — p = 0.303: there is no evidence against MCAR.

**Implicit/behavioural (Combined packages)** (n = 186) — p = 0.694: there is no evidence against MCAR.

None of the three subsets rejects MCAR at the 5% level, so the formal test does not confirm the earlier qualitative concern that ``Implicit_Score`` and ``Repeat_Purchase_Pct`` are MAR/MNAR. This should be read with the usual caveats — Little's test has limited power, assumes multivariate normality, and failing to reject MCAR is not proof of MCAR (see Schouten et al., 2021, and the ``pyampute`` documentation). Crucially, the *overall* amount of truly missing data is very small (1.56% of all cells), which is well below the ≈5% threshold below which multiple imputation is generally considered unnecessary (Dettori et al., 2018).

### Handling recommendation
Given (i) the very small overall missingness — 1.56% of cells, well under the ≈5% rule of thumb in Dettori et al. (2018) — and (ii) no evidence against MCAR from Little's test, **multiple imputation is not appropriate** here. However, *full* listwise deletion across all 18 columns would drop 184 of 750 rows (24.5%), because the 211 missing cells are spread over 8 columns. The recommended approach is therefore **per-model deletion**: drop rows missing in only the columns a given model actually uses (``data_utilities.drop_missing_rows`` with an explicit column subset).

The savings are substantial for the core research model — dropping rows missing in the five core variables removes only 36 rows (4.8%) rather than 184. Models centred on the launch metrics remain the costly case: within the 403 launched concepts, dropping rows missing in any launch metric removes 111 (27.5%), reflecting the concentration of missingness in those fields. Each model should therefore specify exactly the columns it uses, so missingness in irrelevant columns does not cause data loss.

