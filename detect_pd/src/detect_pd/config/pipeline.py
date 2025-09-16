"""Composite configuration describing the entire pipeline."""
from __future__ import annotations

from pydantic import Field

from .base import BaseConfig
from .data_ingestion import DataIngestionConfig
from .evaluation import EvaluationConfig
from .feature_engineering import FeatureEngineeringConfig
from .model import ModelTrainingConfig
from .preprocessing import PreprocessingConfig
from .split import SplitConfig
from .tracking import TrackingConfig


class PipelineConfig(BaseConfig):
    """Aggregated configuration object encapsulating all pipeline step configs."""

    data_ingestion: DataIngestionConfig = Field(..., description="Configuration for data ingestion step.")
    split: SplitConfig = Field(..., description="Dataset split configuration.")
    preprocessing: PreprocessingConfig = Field(
        ..., description="Preprocessing and feature derivation configuration."
    )
    feature_engineering: FeatureEngineeringConfig = Field(
        ..., description="Configuration for feature selection step."
    )
    model_training: ModelTrainingConfig = Field(
        ..., description="Model training configuration." 
    )
    evaluation: EvaluationConfig = Field(
        ..., description="Evaluation metrics and plotting configuration."
    )
    tracking: TrackingConfig = Field(
        default_factory=TrackingConfig,
        description="MLflow tracking configuration with sensible defaults.",
    )
