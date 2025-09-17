"""Configuration for feature engineering steps."""
from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import Field

from .base import BaseConfig


class FeatureSelectionTargetConfig(BaseConfig):
    """Configuration for an individual LASSO feature selection target."""

    target_name: str | None = Field(
        None, description="Name of the target column to use for this configuration."
    )
    problem_type: Literal["binary", "regression"] = Field(
        "regression", description="Treat target as regression or binary classification."
    )
    threshold: float | None = Field(
        None,
        description="Optional threshold for deriving a binary adequacy label (e.g. Kt/V >= 1.7).",
    )
    alpha: float = Field(0.01, gt=0, description="LASSO regularisation strength.")
    max_iter: int = Field(1000, gt=0, description="Maximum iterations for solver convergence.")
    min_features: int = Field(
        5,
        ge=1,
        description="Minimum number of features to retain regardless of sparsity result.",
    )


class FeatureEngineeringConfig(BaseConfig):
    """Top-level configuration controlling feature engineering."""

    targets: Dict[str, FeatureSelectionTargetConfig] = Field(
        ..., description="Mapping of target names to their feature selection configuration."
    )
    shared_allowed_features: List[str] = Field(
        default_factory=list,
        description="Feature names that must always be kept in the dataset.",
    )
