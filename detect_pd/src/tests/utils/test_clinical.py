"""Unit tests for clinical utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from detect_pd.utils import (
    CharlsonConfig,
    compute_bmi,
    compute_bsa_du_bois,
    compute_charlson_index,
    derive_time_features,
    validate_prediction_ranges,
)


def test_compute_bmi_round_trip():
    bmi = compute_bmi(weight_kg=70, height_cm=175)
    assert round(bmi, 2) == 22.86


def test_compute_bsa_du_bois():
    bsa = compute_bsa_du_bois(weight_kg=70, height_cm=175)
    assert round(bsa, 4) == 1.8481


def test_compute_charlson_index_with_age_and_renal():
    comorbidities = ["diabetes", "myocardial_infarction"]
    score = compute_charlson_index(comorbidities, age=65)
    # diabetes=1, MI=1, renal disease=2, age bracket (60-69)=2 -> total 6
    assert score == 6


def test_compute_charlson_index_custom_weights():
    config = CharlsonConfig(weights={"custom": 3}, include_renal_disease=False)
    score = compute_charlson_index(["custom"], age=None, config=config)
    assert score == 3


def test_derive_time_features():
    df = pd.DataFrame(
        {
            "egfr": pd.to_datetime(["2023-01-01", "2023-01-10"]),
            "pd_start": pd.to_datetime(["2023-01-11", "2023-01-20"]),
            "tki": pd.to_datetime(["2022-12-20", "2023-01-05"]),
            "assessment": pd.to_datetime(["2023-02-10", "2023-02-15"]),
        }
    )
    features = derive_time_features(
        df,
        {
            "egfr_below_10_date": "egfr",
            "pd_start_date": "pd_start",
            "tki_date": "tki",
            "assessment_date": "assessment",
        },
    )
    assert features["failure_period_days"].iloc[0] == 10
    assert features["waiting_period_days"].iloc[1] == 15
    assert features["pd_period_days"].iloc[0] == 30


def test_validate_prediction_ranges():
    ktv_valid = np.array([1.2, 2.5, 3.8])
    assert validate_prediction_ranges(ktv_valid, "ktv")

    pet_invalid = np.array([0.05, 0.9])
    assert not validate_prediction_ranges(pet_invalid, "pet")
