"""Tests for the data ingestion helper."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from detect_pd.config import DataIngestionConfig
from detect_pd.steps.data_ingestion import ingest_data_frame


def _build_sample_dataframe() -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Unnamed: 0_level_0", "UUID"),
            ("Demographics", "Age"),
            ("Demographics", "Gender"),
            ("PD related", "Date of PD start"),
            ("Ground Truth", "Kt/V"),
        ]
    )
    return pd.DataFrame(
        [
            ["patient-1", 60, "F", "2023-01-10", 1.8],
            ["patient-2", 55, "M", "2023-02-15", None],
        ],
        columns=columns,
    )


def _build_config(file_path: Path) -> DataIngestionConfig:
    return DataIngestionConfig(
        file_path=file_path,
        sheet_name="Sheet1",
        header_rows=[0, 1],
        required_columns=["patient_id", "age", "ktv", "pd_start_date"],
        date_columns=["pd_start_date"],
        column_renames={
            "Unnamed: 0_level_0::UUID": "patient_id",
            "Demographics::Age": "age",
            "PD related::Date of PD start": "pd_start_date",
            "Ground Truth::Kt/V": "ktv",
        },
        drop_columns=["Demographics::Gender"],
        index_column="patient_id",
        numeric_validation_rules={"ktv": (0.5, 4.0)},
    )


def test_ingest_data_frame_drops_missing_outcomes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sample_df = _build_sample_dataframe()
    monkeypatch.setattr(
        "detect_pd.steps.data_ingestion.pd.read_excel",
        lambda *args, **kwargs: sample_df,
    )

    config = _build_config(tmp_path / "dummy.xlsx")
    df = ingest_data_frame(config)

    assert df.index.name == "patient_id"
    assert list(df.columns) == ["age", "pd_start_date", "ktv"]
    assert len(df) == 1
    assert df.iloc[0]["age"] == 60
    assert pd.api.types.is_datetime64_any_dtype(df["pd_start_date"]) is True


def test_ingest_data_frame_numeric_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df_invalid = pd.DataFrame(
        [["patient-1", 5.5]],
        columns=pd.MultiIndex.from_tuples(
            [
                ("Unnamed: 0_level_0", "UUID"),
                ("Ground Truth", "Kt/V"),
            ]
        ),
    )
    monkeypatch.setattr(
        "detect_pd.steps.data_ingestion.pd.read_excel",
        lambda *args, **kwargs: df_invalid,
    )

    config = DataIngestionConfig(
        file_path=tmp_path / "invalid.xlsx",
        sheet_name="Sheet1",
        header_rows=[0, 1],
        required_columns=["patient_id", "ktv"],
        column_renames={
            "Unnamed: 0_level_0::UUID": "patient_id",
            "Ground Truth::Kt/V": "ktv",
        },
        numeric_validation_rules={"ktv": (0.5, 4.0)},
    )

    with pytest.raises(ValueError):
        ingest_data_frame(config)
