"""Configuration schema for model training."""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import Field, root_validator

from .base import BaseConfig


class ModelDefinition(BaseConfig):
    """Describes a single model instance to be trained."""

    model_type: Literal["xgboost", "random_forest", "linear_regression"] = Field(
        ..., description="Which algorithm to train."
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters passed to the estimator constructor."
    )
    cross_validation_folds: int = Field(
        5, ge=2, description="Number of folds to use when performing cross-validation."
    )
    early_stopping_rounds: int | None = Field(
        None, ge=1, description="Early stopping rounds for boosting models when applicable."
    )
    use_mlflow_autolog: bool = Field(
        True, description="Enable MLflow autologging for this model run."
    )
    eval_metric: str | None = Field(
        None, description="Primary evaluation metric used during training/validation."
    )

    @root_validator
    def _validate_early_stopping(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        model_type = values.get("model_type")
        early_stopping = values.get("early_stopping_rounds")
        if early_stopping is not None and model_type != "xgboost":
            raise ValueError("early_stopping_rounds is only supported for XGBoost models.")
        return values


class TargetModelCollection(BaseConfig):
    """Collection of model definitions for a specific prediction target."""

    target: str = Field(..., description="Name of the outcome the models train on.")
    models: List[ModelDefinition] = Field(
        ..., min_items=1, description="List of candidate models to train for the target."
    )
    primary_metric: str = Field(
        "r2", description="Metric used to determine the champion model for this target."
    )


class ModelTrainingConfig(BaseConfig):
    """Top-level configuration for all model training tasks."""

    targets: List[TargetModelCollection] = Field(
        ..., min_items=1, description="Per-target model configuration objects."
    )
    n_jobs: int = Field(
        1, ge=1, description="Parallelism level for algorithms supporting n_jobs."
    )
    persist_models: bool = Field(
        True, description="Whether to persist trained model artefacts to disk/MLflow." 
    )
