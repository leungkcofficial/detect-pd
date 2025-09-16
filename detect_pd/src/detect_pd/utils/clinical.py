"""Clinical calculation utilities for the DETECT-PD project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

CCI_BASE_WEIGHTS: Mapping[str, int] = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "chronic_pulmonary_disease": 1,
    "connective_tissue_disease": 1,
    "peptic_ulcer_disease": 1,
    "mild_liver_disease": 1,
    "diabetes": 1,
    "diabetes_with_end_organ_damage": 2,
    "hemiplegia": 2,
    "moderate_or_severe_renal_disease": 2,
    "any_tumor": 2,
    "leukemia": 2,
    "lymphoma": 2,
    "moderate_or_severe_liver_disease": 3,
    "metastatic_solid_tumor": 6,
    "aids_hiv": 6,
}


def compute_bmi(weight_kg: float, height_cm: float) -> float:
    """Compute body mass index in kg/m^2."""

    if height_cm <= 0:
        raise ValueError("Height must be positive when computing BMI.")
    height_m = height_cm / 100.0
    return float(weight_kg / (height_m**2))


def compute_bsa_du_bois(weight_kg: float, height_cm: float) -> float:
    """Compute body surface area using the Du Bois formula."""

    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("Weight and height must be positive for BSA calculation.")
    return float(0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725))


def age_adjustment(age: float) -> int:
    """Return standard Charlson age adjustment based on age brackets."""

    if age < 50:
        return 0
    if age < 60:
        return 1
    if age < 70:
        return 2
    if age < 80:
        return 3
    if age < 90:
        return 4
    return 5


@dataclass(frozen=True)
class CharlsonConfig:
    """Configuration parameters for Charlson Comorbidity Index calculation."""

    weights: Mapping[str, int] = None
    include_renal_disease: bool = True
    include_age: bool = True

    def resolve_weight(self, comorbidity: str) -> int:
        weights = self.weights or CCI_BASE_WEIGHTS
        return weights.get(comorbidity, 0)


def compute_charlson_index(
    comorbidities: Iterable[str],
    age: Optional[float],
    config: CharlsonConfig | None = None,
) -> int:
    """Compute the Charlson Comorbidity Index for a patient."""

    config = config or CharlsonConfig()
    score = 0
    for comorbidity in comorbidities:
        score += config.resolve_weight(comorbidity)
    if config.include_renal_disease:
        score += config.resolve_weight("moderate_or_severe_renal_disease")
    if config.include_age and age is not None:
        score += age_adjustment(age)
    return score


def compute_time_interval(start: pd.Series, end: pd.Series) -> pd.Series:
    """Compute the time interval in days between two datetime series."""

    delta = end - start
    return delta.dt.days.astype("float")


def derive_time_features(df: pd.DataFrame, column_map: Mapping[str, str]) -> pd.DataFrame:
    """Create failure/waiting/PD period features from raw date columns."""

    required_keys = {"egfr_below_10_date", "pd_start_date", "tki_date", "assessment_date"}
    missing = required_keys - set(column_map)
    if missing:
        raise KeyError(f"Missing mappings for required time features: {missing}")

    egfr_col = column_map["egfr_below_10_date"]
    pd_start_col = column_map["pd_start_date"]
    tki_col = column_map["tki_date"]
    assessment_col = column_map["assessment_date"]

    features = pd.DataFrame(index=df.index)
    features["failure_period_days"] = compute_time_interval(df[egfr_col], df[pd_start_col])
    features["waiting_period_days"] = compute_time_interval(df[tki_col], df[pd_start_col])
    features["pd_period_days"] = compute_time_interval(df[pd_start_col], df[assessment_col])
    return features


def validate_prediction_ranges(predictions: np.ndarray, target: str) -> bool:
    """Ensure predictions fall within expected clinical ranges."""

    if target == "ktv":
        return bool(np.all((predictions >= 0.5) & (predictions <= 4.0)))
    if target == "pet":
        return bool(np.all((predictions >= 0.1) & (predictions <= 1.0)))
    raise ValueError(f"Unknown target '{target}' provided for validation.")
