"""Unit tests for dataframe utilities."""
from __future__ import annotations

import pandas as pd

from detect_pd.utils import flatten_multiindex_columns, parse_dates, rename_and_select_columns


def test_flatten_multiindex_columns():
    arrays = [
        ["Demographics", "Demographics"],
        ["Age", "Gender"],
    ]
    multi_columns = pd.MultiIndex.from_arrays(arrays)
    df = pd.DataFrame([[65, "F"]], columns=multi_columns)
    flattened = flatten_multiindex_columns(df)
    assert list(flattened.columns) == ["Demographics::Age", "Demographics::Gender"]


def test_flatten_regular_columns_returns_stripped_strings():
    df = pd.DataFrame(columns=["  Age ", " Height "])
    flattened = flatten_multiindex_columns(df)
    assert list(flattened.columns) == ["Age", "Height"]


def test_rename_and_select_columns_validates_required():
    df = pd.DataFrame({"old": [1], "drop": [2]})
    renamed = rename_and_select_columns(
        df,
        renames={"old": "new"},
        drop_columns=["drop"],
        required_columns=["new"],
    )
    assert list(renamed.columns) == ["new"]


def test_parse_dates():
    df = pd.DataFrame({"date": ["2023-01-01", "invalid"]})
    parsed = parse_dates(df, ["date"])
    assert pd.isna(parsed.loc[1, "date"])
    assert parsed.loc[0, "date"].year == 2023
