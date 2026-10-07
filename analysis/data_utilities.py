"""Shared data loading and cleaning utilities.

This module is the single source of truth for reading and preparing the
project's raw CSV datasets. Analysis scripts should import their loaders from
here instead of reading the CSVs directly, so that parsing and cleaning rules
stay consistent across the whole project.

The raw files carry a UTF-8 BOM and store money as ``€``-formatted strings
(e.g. ``"€13,940.00"``); this module strips the BOM, parses those strings into
floats, coerces the numeric columns, and assigns sensible dtypes (nullable
integers/floats and categoricals) while preserving missing values. Values that
are legitimately absent ("not applicable") are marked with the string ``"NA"``
so they can be told apart from genuinely missing values (``pd.NA``). An
optional listwise-deletion step (:func:`drop_missing_rows`) removes rows with
genuinely missing values while leaving the ``"NA"`` markers intact.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

HISTORICAL_DATA_FILENAME = "historical_data.csv"
DATA_DICTIONARY_FILENAME = "data_dictionary.csv"

# ---------------------------------------------------------------------------
# Column metadata for historical_data.csv
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS: list[str] = [
    "Category",
    "Innovation_Type",
    "Research_Package",
    "Recommendation",
]

INTEGER_COLUMNS: list[str] = ["Test_Year", "Sample_Size", "Turnaround_Days"]

BINARY_COLUMNS: list[str] = ["Launched"]

CURRENCY_COLUMNS: list[str] = ["Research_Cost_EUR", "Launch_Support_EUR"]

FLOAT_COLUMNS: list[str] = [
    "Stated_Appeal",
    "Purchase_Intent",
    "Behavioural_Choice_Pct",
    "Implicit_Score",
    "Distribution_Pct",
    "Sales_vs_Target_Pct",
    "Repeat_Purchase_Pct",
]

TEXT_COLUMNS: list[str] = ["Concept_ID"]

#: Marker used for values that are legitimately absent ("not applicable"),
#: distinct from genuinely missing values (``pd.NA``).
NA_MARKER: str = "NA"

#: Columns that are only applicable under a condition. Their legitimately
#: absent ("not applicable") values are replaced with :data:`NA_MARKER` during
#: cleaning so they can be told apart from genuinely missing values.
CONDITIONAL_COLUMNS: list[str] = [
    "Behavioural_Choice_Pct",
    "Implicit_Score",
    "Launch_Support_EUR",
    "Distribution_Pct",
    "Sales_vs_Target_Pct",
    "Repeat_Purchase_Pct",
]


def _not_applicable(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean mask of cells whose value is legitimately absent.

    A cell is "not applicable" when the concept's design means the metric was
    never meant to be collected (e.g. launch metrics for unlaunched concepts).
    """
    if column == "Behavioural_Choice_Pct":
        return ~df["Research_Package"].isin(["Behavioural", "Combined"])
    if column == "Implicit_Score":
        return df["Research_Package"] != "Combined"
    if column in (
        "Launch_Support_EUR",
        "Distribution_Pct",
        "Sales_vs_Target_Pct",
        "Repeat_Purchase_Pct",
    ):
        return df["Launched"] != 1
    return pd.Series(False, index=df.index)


def get_data_path(filename: str) -> Path:
    """Return the absolute path to ``filename`` inside the ``data/`` directory."""
    return DATA_DIR / filename


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV with the project's standard defaults.

    The source files carry a UTF-8 BOM, so they are decoded with ``utf-8-sig``.
    Empty cells are treated as missing values (pandas default).
    """
    defaults: dict[str, Any] = {"encoding": "utf-8-sig"}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def parse_currency(value: Any) -> float | None:
    """Convert a ``€``-formatted string such as ``"€13,940.00"`` to a float.

    Returns ``None`` for missing/blank values so the result can be stored in a
    nullable float column. Already-numeric values are passed through unchanged.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    text = (
        str(value)
        .replace("€", "")
        .replace("\u20ac", "")
        .replace(",", "")
        .replace("\u00a0", "")
        .strip()
    )
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and type a freshly loaded historical dataset.

    Returns a copy of ``df`` with:

    * column names trimmed of whitespace,
    * ``€``-formatted currency columns parsed into nullable floats,
    * numeric columns coerced to nullable ``Float64`` / ``Int64``,
    * categorical columns cast to ``category`` (with whitespace trimmed),
    * the ``Concept_ID`` text column trimmed and stored as ``string``,
    * "not applicable" values replaced with the ``"NA"`` marker so they are
      distinguishable from genuinely missing values (``pd.NA``).
    """
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    for column in CURRENCY_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(parse_currency).astype("Float64")

    for column in FLOAT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Float64")

    for column in INTEGER_COLUMNS + BINARY_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = (
                df[column].astype("string").str.strip().replace({"": pd.NA}).astype("category")
            )

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    # Replace "not applicable" (absent by design) values with the "NA" marker
    # so they are distinguishable from genuinely missing values (pd.NA). Cast
    # to object first so the column can hold both numbers and the "NA" string.
    for column in CONDITIONAL_COLUMNS:
        if column in df.columns:
            not_applicable = _not_applicable(df, column)
            df[column] = df[column].astype(object).mask(not_applicable, NA_MARKER)

    return df


def drop_missing_rows(
    df: pd.DataFrame, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Drop rows with genuinely missing values in ``columns`` (listwise deletion).

    Only truly missing values (``pd.NA``) are treated as missing; structural
    ``"NA"`` (absent by design) markers are valid and are left intact.

    * ``columns=None`` — full listwise deletion: drop rows missing in *any*
      column.
    * ``columns=[...]`` — per-model deletion: drop rows only when one of the
      listed columns is missing, leaving missingness in other columns intact.

    Returns a new DataFrame.
    """
    if columns is None:
        return df.dropna()
    columns = list(columns)
    if not columns:
        return df.copy()
    return df.dropna(subset=columns)


def load_historical_data(
    path: str | Path | None = None,
    *,
    clean: bool = True,
    drop_missing: bool | Sequence[str] = False,
) -> pd.DataFrame:
    """Load (and, by default, clean) the historical concept-test dataset.

    ``drop_missing`` controls listwise deletion after cleaning:

    * ``False`` (default) — no rows are dropped.
    * ``True`` — full listwise deletion (rows missing in any column).
    * a sequence of column names — per-model deletion (rows missing in any of
      those columns only).

    Only truly missing values (``pd.NA``) are treated as missing; structural
    ``"NA"`` (absent by design) values are never deleted. Deletion relies on
    the ``"NA"`` markers, so cleaning is implicitly enabled whenever
    ``drop_missing`` is not ``False``.
    """
    source = Path(path) if path is not None else get_data_path(HISTORICAL_DATA_FILENAME)
    df = read_csv(source)
    if clean or drop_missing is not False:
        df = clean_historical_data(df)
    if drop_missing is True:
        df = drop_missing_rows(df)
    elif drop_missing is not False:
        df = drop_missing_rows(df, columns=drop_missing)
    return df


def load_data_dictionary(path: str | Path | None = None) -> pd.DataFrame:
    """Load the data dictionary (column-level metadata) with whitespace trimmed."""
    source = Path(path) if path is not None else get_data_path(DATA_DICTIONARY_FILENAME)
    df = read_csv(source)
    df.columns = [str(column).strip() for column in df.columns]
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype("string").str.strip().replace({"": pd.NA})
    return df


__all__ = [
    "CONDITIONAL_COLUMNS",
    "DATA_DIR",
    "NA_MARKER",
    "PROJECT_ROOT",
    "clean_historical_data",
    "drop_missing_rows",
    "get_data_path",
    "load_data_dictionary",
    "load_historical_data",
    "parse_currency",
    "read_csv",
]


if __name__ == "__main__":
    historical = load_historical_data()
    dictionary = load_data_dictionary()

    print("Historical data shape:", historical.shape)
    print("\nDtypes:\n", historical.dtypes)
    print("\nMissing values per column:\n", historical.isna().sum())
    print("\nHead:\n", historical.head())
    print("\nData dictionary:\n", dictionary)
