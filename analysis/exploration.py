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

#: Nominal categorical columns tabulated as frequencies (count + percentage).
CATEGORICAL_COLUMNS: list[tuple[str, str]] = [
    ("Category", "Product category"),
    ("Innovation_Type", "Innovation type"),
    ("Research_Package", "Research package"),
    ("Recommendation", "Research recommendation"),
]


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

    print(f"Exploration log written to: {log.path}")


if __name__ == "__main__":
    main()
