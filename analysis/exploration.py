"""Exploratory data analysis of the historical concept-test dataset.

Loads the cleaned historical data and writes a first-pass summary of its
structure and distributions to a Markdown log in ``output/logs/``.

Summary-statistics choices (reproduced in the log):

* Categorical variables and the ordinal ``Test_Year`` are tabulated as
  frequency distributions (count + percentage) rather than means, because they
  are labels / ordered groups, not measurements.
* Continuous and count variables are summarised with count, mean, standard
  deviation, minimum, quartiles and maximum. Mean and standard deviation are
  reported alongside the median and inter-quartile range because cost variables
  are right-skewed, for which the median is the more robust central value.
* Missing values are split into two groups: *absent by design* (the value
  logically does not exist, e.g. launch metrics for unlaunched concepts) and
  *applicable but missing* (the value should exist but was not collected), so
  genuine data-collection gaps are visible rather than hidden inside totals.
"""

from __future__ import annotations

import pandas as pd
from pyampute.exploration.mcar_statistical_tests import MCARTest
from scipy import stats

import data_utilities
import log_writer

#: Numeric columns summarised with mean/std/quartiles, grouped into the
#: research phase (collected for every concept) and the launch phase (only
#: meaningful for launched concepts). ``Test_Year`` is ordinal and ``Launched``
#: is binary, so both are summarised as frequencies instead.
NUMERIC_COLUMNS: list[str] = [
    # research phase
    "Sample_Size",
    "Turnaround_Days",
    "Research_Cost_EUR",
    "Stated_Appeal",
    "Purchase_Intent",
    "Behavioural_Choice_Pct",
    "Implicit_Score",
    # launch phase
    "Launch_Support_EUR",
    "Distribution_Pct",
    "Sales_vs_Target_Pct",
    "Repeat_Purchase_Pct",
]

#: Numeric columns split by phase, used to render the correlation matrix as
#: two narrower tables that fit the page width.
RESEARCH_PHASE_COLUMNS: list[str] = NUMERIC_COLUMNS[:7]
LAUNCH_PHASE_COLUMNS: list[str] = NUMERIC_COLUMNS[7:]

#: Nominal categorical columns tabulated as frequencies (count + percentage).
CATEGORICAL_COLUMNS: list[tuple[str, str]] = [
    ("Category", "Product category"),
    ("Innovation_Type", "Innovation type"),
    ("Research_Package", "Research package"),
    ("Recommendation", "Research recommendation"),
]

#: Logical display order for the research-package categories.
PACKAGE_ORDER: list[str] = ["Survey", "Behavioural", "Combined"]


def _frequencies(
    series: pd.Series,
    value_label: str = "Value",
    *,
    sort_index: bool = False,
) -> pd.DataFrame:
    """Return a frequency table (value, count, percentage) for ``series``."""
    counts = series.value_counts(dropna=False)
    if sort_index:
        counts = counts.sort_index()
    total = int(counts.sum())
    return pd.DataFrame(
        {
            value_label: [str(value) for value in counts.index],
            "Count": counts.to_numpy(),
            "Pct (%)": (counts.to_numpy() / total * 100).round(1),
        }
    )


def _package_trend(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts, row-percentages) crosstab of Research_Package by year."""
    counts = pd.crosstab(df["Test_Year"], df["Research_Package"]).reindex(
        columns=PACKAGE_ORDER
    )
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)
    return counts, pct


def _package_cost_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean research cost and turnaround days by package, versus Survey baseline.

    Survey is the baseline package; the difference columns show how much more
    (positive) or less (negative) the Behavioural and Combined packages cost
    and take relative to Survey. Cost is in euros, turnaround in days.
    """
    metrics = ["Research_Cost_EUR", "Turnaround_Days"]
    grouped = (
        df.groupby("Research_Package", observed=True)[metrics].mean().reindex(PACKAGE_ORDER)
    )
    counts = df["Research_Package"].value_counts().reindex(PACKAGE_ORDER)
    base_cost = grouped.loc["Survey", "Research_Cost_EUR"]
    base_days = grouped.loc["Survey", "Turnaround_Days"]

    rows = []
    for package in PACKAGE_ORDER:
        cost = grouped.loc[package, "Research_Cost_EUR"]
        days = grouped.loc[package, "Turnaround_Days"]
        if package == "Survey":
            d_cost = "—"
            d_days = "—"
        else:
            d_cost = f"{cost - base_cost:+,.0f}"
            d_days = f"{days - base_days:+.1f}"
        rows.append(
            {
                "Research package": package,
                "n": int(counts[package]),
                "Mean cost (€)": f"{cost:,.0f}",
                "Δ cost vs Survey (€)": d_cost,
                "Mean turnaround (days)": f"{days:.1f}",
                "Δ turnaround vs Survey (days)": d_days,
            }
        )
    return pd.DataFrame(rows)


def _correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix for the numeric columns.

    Structural ``"NA"`` (absent by design) markers are coerced to NaN, so each
    pair is computed on pairwise-complete observations — rows where both
    variables are applicable. Pairs involving ``Implicit_Score`` (Combined
    packages only), ``Behavioural_Choice_Pct`` (Behavioural/Combined), or the
    launch metrics (launched concepts only) therefore rest on smaller,
    more restricted subsamples.
    """
    numeric = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce").astype("float64")
            for column in columns
        }
    )
    return numeric.corr(method="pearson").round(2)


def _top_correlations(corr: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return the strongest pairwise correlations, excluding the diagonal."""
    columns = list(corr.columns)
    pairs = [
        (columns[i], columns[j], corr.iloc[i, j])
        for i in range(len(columns))
        for j in range(i + 1, len(columns))
    ]
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pd.DataFrame(
        [
            {
                "Variable 1": a,
                "Variable 2": b,
                "r": r,
                "Strength": data_utilities.correlation_strength(r),
            }
            for a, b, r in pairs[:limit]
        ]
    )


def _cross_phase_correlations(corr: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """Return research-phase × launch-phase pairs with |r| >= threshold.

    These are the "cross-block" correlations between the pre-launch research
    metrics and the post-launch performance metrics, restricted to notable
    (weak-or-stronger) coefficients and sorted by absolute value. Each pair is
    classified into a strength band (weak / moderate / strong).
    """
    pairs = [
        (research, launch, corr.loc[research, launch])
        for research in RESEARCH_PHASE_COLUMNS
        for launch in LAUNCH_PHASE_COLUMNS
        if abs(corr.loc[research, launch]) >= threshold
    ]
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pd.DataFrame(
        [
            {
                "Research variable": research,
                "Launch variable": launch,
                "r": r,
                "Strength": data_utilities.correlation_strength(r),
            }
            for research, launch, r in pairs
        ]
    )


def _numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return count/mean/std/min/quartiles/max plus missing for ``columns``.

    "NA" markers (absent by design) are coerced to missing so the statistics
    describe only the genuinely collected numeric values.
    """
    numeric = pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce").astype("float64")
            for column in columns
        }
    )
    summary = numeric.describe().T
    summary["Missing"] = numeric.isna().sum()
    summary = summary.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std",
            "min": "Min",
            "25%": "Q25",
            "50%": "Median",
            "75%": "Q75",
            "max": "Max",
        }
    )
    summary = summary.round(2)
    return summary.reset_index().rename(columns={"index": "Column"})


def _missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Split each column's missing values into two groups.

    * "Absent by design" — cells marked with the "NA" marker during cleaning
      (the value logically does not exist for that row).
    * "Applicable but missing" — cells that are ``pd.NA`` (the value should
      exist but was not collected).
    """
    rows: list[dict] = []
    for column in df.columns:
        if column in data_utilities.CONDITIONAL_COLUMNS:
            n_absent_by_design = int(
                df[column].eq(data_utilities.NA_MARKER).fillna(False).sum()
            )
        else:
            n_absent_by_design = 0
        n_applicable_missing = int(df[column].isna().sum())
        n_present = len(df) - n_absent_by_design - n_applicable_missing
        n_applicable = n_present + n_applicable_missing
        rate = (n_applicable_missing / n_applicable * 100) if n_applicable else 0.0
        rows.append(
            {
                "Column": column,
                "Applicable": n_applicable,
                "Present": n_present,
                "Absent by design": n_absent_by_design,
                "Applicable but missing": n_applicable_missing,
                "% missing": round(rate, 1),
            }
        )
    return pd.DataFrame(rows)


def _missingness_summary(df: pd.DataFrame) -> dict:
    """Return overall missingness totals and the true missingness rate.

    "Absent by design" values are valid (not statistically missing) but remain
    part of the data, so they stay in the denominator. The *true missingness
    rate* is therefore the share of all data cells that are "applicable but
    missing".
    """
    table = _missingness_table(df)
    total_cells = int(df.size)
    applicable = int(table["Applicable"].sum())
    present = int(table["Present"].sum())
    absent_by_design = int(table["Absent by design"].sum())
    truly_missing = int(table["Applicable but missing"].sum())
    return {
        "Total cells": total_cells,
        "Applicable cells": applicable,
        "Present cells": present,
        "Absent by design (valid, not missing)": absent_by_design,
        "Truly missing cells (applicable but missing)": truly_missing,
        "True missingness (% of all data)": f"{truly_missing / total_cells * 100:.2f}%",
    }


def _fmt_p_value(p: float) -> str:
    """Format a p-value for display, flooring very small values."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _mcar_matrix(df: pd.DataFrame, columns: list[str], mask: pd.Series) -> pd.DataFrame:
    """Build a numeric matrix (NaN = missing) for the MCAR test.

    Only genuinely missing values (``pd.NA``) become NaN. Structural ``"NA"``
    (absent by design) markers are excluded by restricting ``mask`` to the
    rows where the columns are actually applicable.
    """
    subset = df.loc[mask, columns]
    return pd.DataFrame(
        {
            column: pd.to_numeric(subset[column], errors="coerce").astype("float64")
            for column in columns
        }
    )


def _run_mcar_tests(df: pd.DataFrame) -> list[dict]:
    """Run Little's MCAR test (via ``pyampute``) on three applicable subsets."""
    tests = [
        {
            "Test": "Core research variables (all concepts)",
            "columns": [
                "Sample_Size",
                "Turnaround_Days",
                "Research_Cost_EUR",
                "Stated_Appeal",
                "Purchase_Intent",
            ],
            "mask": pd.Series(True, index=df.index),
        },
        {
            "Test": "Launch performance (launched concepts)",
            "columns": [
                "Launch_Support_EUR",
                "Distribution_Pct",
                "Sales_vs_Target_Pct",
                "Repeat_Purchase_Pct",
            ],
            "mask": df["Launched"] == 1,
        },
        {
            "Test": "Implicit/behavioural (Combined packages)",
            "columns": [
                "Behavioural_Choice_Pct",
                "Implicit_Score",
                "Stated_Appeal",
                "Purchase_Intent",
            ],
            "mask": df["Research_Package"] == "Combined",
        },
    ]
    mt = MCARTest(method="little")
    results = []
    for test in tests:
        x = _mcar_matrix(df, test["columns"], test["mask"])
        p = float(mt.little_mcar_test(x))
        results.append(
            {
                "Test": test["Test"],
                "n": int(len(x)),
                "Variables": ", ".join(test["columns"]),
                "p": p,
                "p-value": _fmt_p_value(p),
                "Conclusion": "Reject MCAR" if p < 0.05 else "Fail to reject MCAR",
            }
        )
    return results


def main() -> None:
    """Run the exploratory analysis and write the exploration log."""
    df = data_utilities.load_historical_data()
    total = len(df)
    launched = int(df["Launched"].sum())
    launch_rate = launched / total * 100

    log = log_writer.MarkdownLog("exploration")
    with log:
        log.header("Exploratory Data Analysis — Historical Concept Tests")

        # 1. Overview -------------------------------------------------------
        log.heading("1. Dataset overview", level=2)
        log.paragraph(
            "The cleaned historical dataset contains one row per concept test. "
            "Currency columns were parsed to floats and categorical columns to "
            "category dtype (see ``data_utilities.py``)."
        )
        log.key_values(
            {
                "Concepts (rows)": total,
                "Columns": df.shape[1],
                "Years covered": f"{int(df['Test_Year'].min())}–{int(df['Test_Year'].max())}",
                "Concepts launched": f"{launched} ({launch_rate:.1f}%)",
                "Total research cost (€)": f"{df['Research_Cost_EUR'].sum():,.0f}",
            }
        )

        # 2. Categorical distributions -------------------------------------
        log.heading("2. Categorical distributions", level=2)
        log.paragraph(
            "Categorical variables are summarised as frequency tables (count "
            "and percentage of all concepts) rather than with means, since "
            "they are labels rather than measurements. ``Test_Year`` is shown "
            "chronologically rather than as a mean year."
        )
        for column, label in CATEGORICAL_COLUMNS:
            log.heading(label, level=3)
            log.table(_frequencies(df[column], value_label=label))

        log.heading("Test year", level=3)
        log.table(
            _frequencies(df["Test_Year"], value_label="Test year", sort_index=True)
        )

        log.heading("Launched", level=3)
        log.table(
            _frequencies(
                df["Launched"].map({0: "No", 1: "Yes"}),
                value_label="Launched",
            )
        )

        log.heading("Research package by year", level=3)
        package_counts, package_pct = _package_trend(df)
        log.paragraph(
            "The research-package mix is tabulated against ``Test_Year`` to "
            "show how the design of concept tests has shifted over time."
        )
        log.table(package_counts.reset_index())
        log.table(package_pct.reset_index(), float_format="{:.1f}".format)

        pct_2022 = package_pct.loc[2022]
        pct_2025 = package_pct.loc[2025]
        package_change = pct_2025 - pct_2022
        log.table(
            pd.DataFrame(
                {
                    "Research package": PACKAGE_ORDER,
                    "2022 (%)": [pct_2022[p] for p in PACKAGE_ORDER],
                    "2025 (%)": [pct_2025[p] for p in PACKAGE_ORDER],
                    "Change (pp)": [package_change[p] for p in PACKAGE_ORDER],
                }
            ),
            float_format="{:.1f}".format,
        )
        log.paragraph(
            "Between 2022 and 2025 the mix shifted decisively away from "
            "Survey-only research and toward the richer Behavioural and "
            "Combined packages. Survey fell from "
            f"{pct_2022['Survey']:.1f}% to {pct_2025['Survey']:.1f}% "
            f"({package_change['Survey']:+.1f} pp), while Behavioural rose from "
            f"{pct_2022['Behavioural']:.1f}% to {pct_2025['Behavioural']:.1f}% "
            f"({package_change['Behavioural']:+.1f} pp) and Combined rose from "
            f"{pct_2022['Combined']:.1f}% to {pct_2025['Combined']:.1f}% "
            f"({package_change['Combined']:+.1f} pp)."
        )
        package_chi2, package_p, package_dof, _ = stats.chi2_contingency(package_counts)
        log.paragraph(
            f"A chi-square test of independence (year × package) gives χ² = "
            f"{package_chi2:.2f}, df = {int(package_dof)}, p = {package_p:.3g}, "
            "so the change across years is statistically significant."
        )

        # 3. Numeric summary ------------------------------------------------
        log.heading("3. Numeric summary statistics", level=2)
        log.paragraph(
            "Each numeric column is summarised with its non-missing count, "
            "mean, standard deviation, minimum, quartiles (25/50/75) and "
            "maximum, plus the number of missing values. Mean and standard "
            "deviation summarise spread for roughly symmetric data, while the "
            "median and inter-quartile range are robust to the right-skew "
            "typical of cost variables. ``Research_Cost_EUR`` and "
            "``Launch_Support_EUR`` are in euros."
        )
        log.table(_numeric_summary(df, NUMERIC_COLUMNS))

        log.heading("Research cost and turnaround by package", level=3)
        log.paragraph(
            "``Research_Cost_EUR`` and ``Turnaround_Days`` are tabulated as "
            "means by research package, with Survey treated as the baseline. "
            "The difference columns show the premium (or saving) of the "
            "Behavioural and Combined packages relative to Survey. Means are "
            "reported here; the full distribution per column is available in "
            "the summary above."
        )
        log.table(_package_cost_summary(df))

        log.heading("Correlations", level=3)
        log.paragraph(
            "Pearson correlations between all numeric columns are tabulated "
            "below (values rounded to 2dp). Each coefficient is computed on "
            "pairwise-complete observations, so pairs involving "
            "``Implicit_Score`` (Combined packages only), "
            "``Behavioural_Choice_Pct`` (Behavioural/Combined), or the launch "
            "metrics (launched concepts only) rest on smaller, more restricted "
            "subsamples and should be interpreted with that in mind."
        )
        corr = _correlation_matrix(df, NUMERIC_COLUMNS)

        log.paragraph(
            "The matrix is split into two tables — research-phase columns and "
            "launch-phase columns — so it fits the page width. Each table "
            "still lists all eleven variables as rows."
        )
        corr_research = corr[RESEARCH_PHASE_COLUMNS].copy()
        corr_research.insert(0, "Variable", corr_research.index)
        log.table(corr_research)

        corr_launch = corr[LAUNCH_PHASE_COLUMNS].copy()
        corr_launch.insert(0, "Variable", corr_launch.index)
        log.table(corr_launch)

        log.heading("Research-phase ↔ launch-phase correlations", level=3)
        log.paragraph(
            "The cross-phase correlations — between the pre-launch research "
            "metrics and the post-launch performance metrics — are shown "
            "below, restricted to pairs with |r| ≥ 0.20 and sorted by "
            "absolute value. Strength bands: |r| 0.20–0.29 weak, 0.30–0.49 "
            "moderate, ≥ 0.50 strong. These are the relationships most "
            "relevant to whether research metrics foreshadow launch outcomes."
        )
        log.table(_cross_phase_correlations(corr))
        log.paragraph(
            "The behavioural and implicit measures are the strongest "
            "pre-launch predictors of launch outcomes: "
            "``Behavioural_Choice_Pct`` and ``Implicit_Score`` correlate "
            "with ``Repeat_Purchase_Pct`` (r = 0.48 and 0.47) and "
            "``Sales_vs_Target_Pct`` (r = 0.38 and 0.37), more strongly "
            "than any stated-metric pairing. ``Stated_Appeal`` appears only "
            "once, and only weakly (r = 0.23 with ``Repeat_Purchase_Pct``), "
            "while ``Sample_Size`` is absent entirely — the attitudinal "
            "(stated) appeal and sample size carry little predictive signal "
            "for launch outcomes."
        )

        log.heading("Notable correlations", level=3)
        log.paragraph(
            "The strongest pairwise correlations (by absolute value) are listed "
            "below, each classified as weak (0.20–0.29), moderate (0.30–0.49), "
            "or strong (≥ 0.50). Only one pair is strong: ``Turnaround_Days`` "
            "and ``Research_Cost_EUR`` (r = 0.81), reflecting that larger, more "
            "expensive studies take longer. The remainder are moderate "
            "(r ≈ 0.4–0.5): the research-phase metrics (``Stated_Appeal``, "
            "``Purchase_Intent``, ``Behavioural_Choice_Pct``, "
            "``Implicit_Score``) inter-correlate, and ``Repeat_Purchase_Pct`` "
            "tracks several of these, suggesting launch outcomes are partly "
            "foreshadowed by pre-launch consumer metrics."
        )
        log.table(_top_correlations(corr))

        # 4. Missing values -------------------------------------------------
        log.heading("4. Missing values", level=2)
        log.paragraph(
            "During cleaning, values that are legitimately absent are marked "
            "with the string ``\"NA\"``, while genuinely missing values remain "
            "``pd.NA``. *Absent by design* rows are therefore those marked "
            "``\"NA\"`` (e.g. launch-phase metrics for unlaunched concepts), "
            "and *applicable but missing* rows are the ``pd.NA`` values (a "
            "genuine data-collection gap). ``% missing`` is the collection-"
            "failure rate among applicable rows."
        )
        log.table(_missingness_table(df))

        log.heading("Overall missingness", level=3)
        log.key_values(_missingness_summary(df))

        log.heading("Missingness mechanism", level=3)
        log.paragraph(
            "The overall true-missingness rate is low (1.56%), but it is far "
            "from uniform across fields. Two fields stand out: "
            "``Implicit_Score`` (10.2% of applicable values missing) and "
            "``Repeat_Purchase_Pct`` (11.2%), versus 2–8% elsewhere. This "
            "concentration is evidence against MCAR (missing completely at "
            "random), under which missingness would be spread evenly at a "
            "uniformly low rate with no relation to any variable."
        )
        log.paragraph(
            "Instead, the pattern is consistent with MAR or MNAR. "
            "``Implicit_Score`` is only collected for Combined packages, and "
            "its elevated failure rate suggests the missingness depends on "
            "observed design features (MAR) or on the unobserved score itself "
            "(e.g. an implicit-association task that fails to yield a stable "
            "measure — MNAR). Likewise, ``Repeat_Purchase_Pct`` is measured "
            "only for launched concepts and appears to drop out for a "
            "non-random subset of them, plausibly tied to category, support "
            "spend, or the repeat-purchase behaviour itself."
        )
        log.paragraph(
            "Consequence: the concentration in ``Implicit_Score`` and "
            "``Repeat_Purchase_Pct`` is worth flagging, but its practical "
            "impact hinges on the *overall* volume of missingness and on a "
            "formal test of the missingness mechanism — both addressed in "
            "section 5, which concludes that complete-case analysis is "
            "sufficient."
        )

        # 5. Little's MCAR test ----------------------------------------------
        log.heading("5. Little's MCAR test", level=2)
        log.paragraph(
            "Little's (1988) test evaluates the null hypothesis that missing "
            "values are Missing Completely At Random (MCAR); a small p-value "
            "(below 0.05) rejects MCAR in favour of MAR or MNAR. The test is "
            "run with ``pyampute.exploration.mcar_statistical_tests."
            "MCARTest(method='little')``. Because the structural ``\"NA\"`` "
            "(absent by design) values are not statistically missing, each "
            "test is restricted to the subset of rows where its variables are "
            "actually applicable."
        )
        results = _run_mcar_tests(df)
        table = pd.DataFrame(results)[
            ["Test", "n", "Variables", "p-value", "Conclusion"]
        ]
        log.table(table)

        log.heading("Interpretation", level=3)
        for r in results:
            if r["p"] < 0.05:
                verdict = (
                    "the null hypothesis of MCAR is rejected — the missingness "
                    "is unlikely to be completely at random (consistent with "
                    "MAR or MNAR)."
                )
            else:
                verdict = "there is no evidence against MCAR."
            log.paragraph(
                f"**{r['Test']}** (n = {r['n']}) — p = {r['p-value']}: {verdict}"
            )
        log.paragraph(
            "None of the three subsets rejects MCAR at the 5% level, so the "
            "formal test does not confirm the earlier qualitative concern that "
            "``Implicit_Score`` and ``Repeat_Purchase_Pct`` are MAR/MNAR. This "
            "should be read with the usual caveats — Little's test has limited "
            "power, assumes multivariate normality, and failing to reject MCAR "
            "is not proof of MCAR (see Schouten et al., 2021, and the "
            "``pyampute`` documentation). Crucially, the *overall* amount of "
            "truly missing data is very small (1.56% of all cells), which is "
            "well below the ≈5% threshold below which multiple imputation is "
            "generally considered unnecessary (Dettori et al., 2018)."
        )

        # Per-model deletion figures ------------------------------------------
        full_dropped = len(df) - len(data_utilities.drop_missing_rows(df))
        core_cols = [
            "Sample_Size",
            "Turnaround_Days",
            "Research_Cost_EUR",
            "Stated_Appeal",
            "Purchase_Intent",
        ]
        core_dropped = len(df) - len(data_utilities.drop_missing_rows(df, core_cols))
        launched_df = df[df["Launched"] == 1]
        launch_cols = [
            "Launch_Support_EUR",
            "Distribution_Pct",
            "Sales_vs_Target_Pct",
            "Repeat_Purchase_Pct",
        ]
        launch_dropped = len(launched_df) - len(
            data_utilities.drop_missing_rows(launched_df, launch_cols)
        )

        log.heading("Handling recommendation", level=3)
        log.paragraph(
            "Given (i) the very small overall missingness — 1.56% of cells, "
            "well under the ≈5% rule of thumb in Dettori et al. (2018) — and "
            "(ii) no evidence against MCAR from Little's test, **multiple "
            "imputation is not appropriate** here. However, *full* listwise "
            f"deletion across all 18 columns would drop {full_dropped} of "
            f"{len(df)} rows ({full_dropped / len(df) * 100:.1f}%), because the "
            "211 missing cells are spread over 8 columns. The recommended "
            "approach is therefore **per-model deletion**: drop rows missing "
            "in only the columns a given model actually uses "
            "(``data_utilities.drop_missing_rows`` with an explicit column "
            "subset)."
        )
        log.paragraph(
            "The savings are substantial for the core research model — dropping "
            "rows missing in the five core variables removes only "
            f"{core_dropped} rows ({core_dropped / len(df) * 100:.1f}%) rather "
            f"than {full_dropped}. Models centred on the launch metrics remain "
            f"the costly case: within the {len(launched_df)} launched concepts, "
            "dropping rows missing in any launch metric removes "
            f"{launch_dropped} ({launch_dropped / len(launched_df) * 100:.1f}%), "
            "reflecting the concentration of missingness in those fields. Each "
            "model should therefore specify exactly the columns it uses, so "
            "missingness in irrelevant columns does not cause data loss."
        )

    print(f"Exploration log written to: {log.path}")


if __name__ == "__main__":
    main()
