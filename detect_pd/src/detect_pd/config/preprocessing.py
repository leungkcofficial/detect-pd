"""Configuration models for preprocessing and feature engineering."""
from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import Field

from .base import BaseConfig


class PreprocessingConfig(BaseConfig):
    """Parameters steering preprocessing transformations."""

    scaling_method: Literal["standard", "minmax"] = Field(
        "standard", description="Scaling strategy for numeric features."
    )
    minmax_feature_range: tuple[float, float] = Field(
        (0.0, 1.0), description="Feature range for MinMax scaling."
    )
    log_transform_features: List[str] = Field(
        default_factory=list,
        description="Numeric features to transform using log1p to reduce skew.",
    )
    categorical_encoding: Literal["one_hot", "label"] = Field(
        "one_hot", description="Encoding strategy for categorical variables."
    )
    imputation_strategy: Literal["mean", "median", "most_frequent", "none"] = Field(
        "none", description="Strategy for imputing missing numeric values."
    )
    include_age_in_cci: bool = Field(
        True, description="Whether to include age-related adjustment in the CCI score."
    )
    cci_weights: Dict[str, int] = Field(
        default_factory=dict,
        description="Override weights for specific comorbidities when computing CCI.",
    )
    bsa_formula: Literal["du_bois"] = Field(
        "du_bois", description="Body surface area formula identifier."
    )
    weight_column: str | None = Field(
        "weight_kg", description="Name of the column containing patient weight in kilograms."
    )
    height_column: str | None = Field(
        "height_cm", description="Name of the column containing patient height in centimetres."
    )
    age_column: str = Field(
        "age", description="Column providing patient age in years for CCI calculation."
    )
    time_column_map: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping for time feature derivation with keys: egfr_below_10_date, "
            "pd_start_date, tki_date, assessment_date."
        ),
    )
    comorbidity_columns: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping from dataset column names to Charlson comorbidity keys."
        ),
    )
    target_columns: List[str] = Field(
        default_factory=lambda: ["ktv", "pet"],
        description="Columns treated as supervised learning targets.",
    )
    categorical_features: List[str] = Field(
        default_factory=list,
        description="Categorical feature columns to encode.",
    )
    numeric_features: List[str] = Field(
        default_factory=list,
        description="Numeric feature columns to scale. If empty, inferred automatically.",
    )
