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
