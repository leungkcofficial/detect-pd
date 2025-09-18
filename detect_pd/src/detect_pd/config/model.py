"""Configuration schema for model training."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .base import BaseConfig


class ModelDefinition(BaseConfig):
    """Describes a single model instance to be trained."""

    model_type: Literal[
        "elastic_net",
        "xgboost",
        "lightgbm",
        "catboost",
        "random_forest",
        "linear_regression",
        "ngboost",
        "quantile_lightgbm",
        "stacked",
    ] = Field(
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
    search_space: Dict[str, Any] = Field(
        default_factory=dict,
        description="Randomized search parameter distributions keyed by hyperparameter name.",
    )
    monotone_constraints: Optional[List[int]] = Field(
        default=None,
        description="Monotonicity constraints passed to compatible models (LightGBM/XGBoost).",
    )
    distribution: Optional[str] = Field(
        default=None,
        description="Distribution identifier for probabilistic models such as NGBoost.",
    )
    quantiles: Optional[List[float]] = Field(
        default=None,
        description="Quantile values to fit for quantile regression models (LightGBM).",
    )
    base_models: Optional[List["ModelDefinition"]] = Field(
        default=None,
        description="Base learner definitions when configuring a stacked ensemble.",
    )

    @model_validator(mode="after")
    def _validate_early_stopping(cls, values: "ModelDefinition") -> "ModelDefinition":
        if values.early_stopping_rounds is not None and values.model_type not in {"xgboost", "lightgbm"}:
            raise ValueError(
                "early_stopping_rounds is only supported for XGBoost/LightGBM models."
            )
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
    random_search_iterations: int = Field(
        20, ge=1, description="Number of parameter samples evaluated during randomized search."
    )
    cv_folds: int = Field(
        5, ge=2, description="Number of folds used for cross-validation during hyperparameter tuning."
    )
    scoring: str = Field(
        "r2", description="Scikit-learn scoring identifier used during hyperparameter tuning."
    )
    random_state: int = Field(
        42, description="Random seed applied to randomized searches and model initialisation where supported."
    )


ModelDefinition.model_rebuild()
TargetModelCollection.model_rebuild()
ModelTrainingConfig.model_rebuild()
