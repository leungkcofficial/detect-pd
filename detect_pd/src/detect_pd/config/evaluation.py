"""Configuration for model evaluation step."""
from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from .base import BaseConfig


class EvaluationThresholdConfig(BaseConfig):
    """Configuration for clinically relevant thresholds used in evaluation."""

    name: str = Field(..., description="Human readable label for the threshold.")
    value: float = Field(..., description="Numeric threshold applied to predictions.")


class EvaluationConfig(BaseConfig):
    """Parameters steering evaluation metrics and artefact generation."""

    metrics: List[str] = Field(
        default_factory=lambda: ["mae", "mse", "r2"],
        description="Metrics to compute on the hold-out test set.",
    )
    comparison_metric: str = Field(
        "r2", description="Metric used for discrimination plots and model ranking."
    )
    calibration_bins: int = Field(
        10, ge=3, description="Number of bins when generating calibration curves."
    )
    thresholds: Dict[str, EvaluationThresholdConfig] = Field(
        default_factory=dict,
        description="Named threshold configurations for adequacy style checks.",
    )
    generate_plots: bool = Field(
        True, description="Whether to generate diagnostic plots during evaluation."
    )
    validate_ranges: bool = Field(
        True, description="Enable clinical plausibility checks on predictions."
    )
    output_dir: str = Field(
        "artifacts/evaluation",
        description="Directory where evaluation reports and plots will be stored.",
    )
