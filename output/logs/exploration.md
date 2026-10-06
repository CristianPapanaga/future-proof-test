# Exploratory Data Analysis — Historical Concept Tests
*Generated 2026-10-06 23:08:24*

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

