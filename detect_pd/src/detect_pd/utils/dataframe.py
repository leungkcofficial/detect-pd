"""Dataframe helper utilities for DETECT-PD ingestion and preprocessing."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd


def flatten_multiindex_columns(df: pd.DataFrame, sep: str = "::") -> pd.DataFrame:
    """Flatten a DataFrame with MultiIndex columns into single-level names."""

    if not isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        return df

    flattened = []
    for col_tuple in df.columns:
        parts = [str(part).strip() for part in col_tuple if part and str(part).strip()]
        flattened.append(sep.join(parts))
    df = df.copy()
    df.columns = flattened
    return df


def rename_and_select_columns(
    df: pd.DataFrame,
    renames: Mapping[str, str],
    drop_columns: Sequence[str],
    required_columns: Iterable[str],
) -> pd.DataFrame:
    """Rename columns, drop unwanted ones, and validate presence of required columns."""

    df = df.copy()
    if renames:
        df.rename(columns=renames, inplace=True)
    if drop_columns:
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after renaming/dropping: {missing}")
    return df


def parse_dates(df: pd.DataFrame, date_columns: Iterable[str]) -> pd.DataFrame:
    """Convert specified columns to datetime while preserving original index."""

    df = df.copy()
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df
