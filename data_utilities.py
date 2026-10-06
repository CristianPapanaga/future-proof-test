"""Shared data loading and cleaning utilities.

This module is the single source of truth for reading and preparing the
project's raw CSV datasets. Analysis scripts should import their loaders from
here instead of reading the CSVs directly, so that parsing and cleaning rules
stay consistent across the whole project.

The raw files carry a UTF-8 BOM and store money as ``€``-formatted strings
(e.g. ``"€13,940.00"``); this module strips the BOM, parses those strings into
floats, coerces the numeric columns, and assigns sensible dtypes (nullable
integers/floats and categoricals) while preserving missing values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
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
    * the ``Concept_ID`` text column trimmed and stored as ``string``.
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

    return df


def load_historical_data(
    path: str | Path | None = None, *, clean: bool = True
) -> pd.DataFrame:
    """Load (and, by default, clean) the historical concept-test dataset."""
    source = Path(path) if path is not None else get_data_path(HISTORICAL_DATA_FILENAME)
    df = read_csv(source)
    return clean_historical_data(df) if clean else df


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
    "DATA_DIR",
    "PROJECT_ROOT",
    "clean_historical_data",
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
