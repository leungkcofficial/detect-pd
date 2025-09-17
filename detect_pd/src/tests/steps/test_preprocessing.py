"""Tests for the preprocessing helper."""
from __future__ import annotations

import numpy as np
import pandas as pd

from detect_pd.config import PreprocessingConfig
from detect_pd.steps.preprocessing import preprocess_dataset


def build_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [60, 45],
            "weight_kg": [70.0, 80.0],
            "height_cm": [170.0, 160.0],
            "history_mi": [1, 0],
            "history_diabetes": [0, 1],
            "egfr_date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "pd_start": pd.to_datetime(["2023-01-15", "2023-02-20"]),
            "tki_date": pd.to_datetime(["2022-12-20", "2023-01-25"]),
            "assessment_date": pd.to_datetime(["2023-03-01", "2023-03-15"]),
            "modality": ["CAPD", "APD"],
            "residual_rf": [400.0, 250.0],
            "ktv": [1.8, 1.4],
            "pet": [0.8, 0.95],
        }
    )


def build_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        scaling_method="standard",
        log_transform_features=["residual_rf"],
        categorical_encoding="one_hot",
        weight_column="weight_kg",
        height_column="height_cm",
        age_column="age",
        time_column_map={
            "egfr_below_10_date": "egfr_date",
            "pd_start_date": "pd_start",
            "tki_date": "tki_date",
            "assessment_date": "assessment_date",
        },
        comorbidity_columns={
            "history_mi": "myocardial_infarction",
            "history_diabetes": "diabetes",
        },
        target_columns=["ktv", "pet"],
        categorical_features=["modality"],
        numeric_features=[],
        include_age_in_cci=True,
    )


def test_preprocess_dataset_generates_expected_features():
    df = build_sample_dataframe()
    config = build_config()

    output = preprocess_dataset(df, config)

    assert {"ktv", "pet"} == set(output.targets.columns)
    # Derived columns present
    for column in [
        "bmi",
        "bsa",
        "charlson_index",
        "failure_period_days",
        "waiting_period_days",
        "pd_period_days",
    ]:
        assert column in output.features.columns

    # One-hot encoding adds modality columns
    encoded_columns = [col for col in output.features.columns if col.startswith("modality_")]
    assert encoded_columns

    # Numeric columns should be scaled to mean approximately zero
    for column in output.artifacts.numeric_columns:
        if column in output.features.columns:
            assert abs(output.features[column].mean()) < 1e-6

    # Charlson index values as expected after reversing scaling
    if output.artifacts.scaler is not None and "charlson_index" in output.artifacts.numeric_columns:
        numeric = output.features[output.artifacts.numeric_columns]
        restored = output.artifacts.scaler.inverse_transform(numeric)
        charlson_index = output.artifacts.numeric_columns.index("charlson_index")
        restored_charlson = restored[:, charlson_index].round(1).tolist()
        assert restored_charlson == [5.0, 3.0]

    assert output.artifacts.scaler is not None
    assert output.artifacts.encoder is not None
    assert set(output.artifacts.encoded_categorical_columns) == set(encoded_columns)
    assert output.artifacts.imputer is None
