"""Model training step for DETECT-PD."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from zenml import step

from detect_pd.config import ModelDefinition, ModelTrainingConfig
from detect_pd.steps.split import SplitOutput
from detect_pd.steps.training_input import ModelTrainingInput

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    r2: float
    mae: float
    mse: float


@dataclass
class ModelResult:
    name: str
    estimator: Any
    metrics: ModelMetrics
    extras: Dict[str, Any]


@dataclass
class TargetTrainingResult:
    target: str
    models: List[ModelResult]


@dataclass
class ModelTrainingResults:
    targets: Dict[str, TargetTrainingResult]


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> ModelMetrics:
    return ModelMetrics(
        r2=r2_score(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        mse=mean_squared_error(y_true, y_pred),
    )


def _apply_monotone_constraints(params: Dict[str, Any], constraints: List[int], framework: str) -> Dict[str, Any]:
    if framework == "xgboost":
        params["monotone_constraints"] = "(" + ",".join(str(c) for c in constraints) + ")"
    elif framework == "lightgbm":
        params["monotone_constraints"] = constraints
    return params


def _instantiate_estimator(model_def: ModelDefinition, feature_columns: List[str]) -> Tuple[Any, Dict[str, Any]]:
    model_type = model_def.model_type
    params = dict(model_def.hyperparameters)
    extras: Dict[str, Any] = {}

    if model_type == "elastic_net":
        estimator = ElasticNet(**params)
    elif model_type == "linear_regression":
        estimator = LinearRegression(**params)
    elif model_type == "random_forest":
        estimator = RandomForestRegressor(n_jobs=params.pop("n_jobs", None) or -1, **params)
    elif model_type == "xgboost":
        from xgboost import XGBRegressor

        params.setdefault("objective", "reg:squarederror")
        params.setdefault("n_estimators", 500)
        params.setdefault("n_jobs", -1)
        if model_def.monotone_constraints:
            params = _apply_monotone_constraints(params, model_def.monotone_constraints, "xgboost")
        estimator = XGBRegressor(**params)
    elif model_type == "lightgbm":
        from lightgbm import LGBMRegressor

        params.setdefault("objective", "regression")
        params.setdefault("n_estimators", 500)
        params.setdefault("n_jobs", -1)
        if model_def.monotone_constraints:
            params = _apply_monotone_constraints(params, model_def.monotone_constraints, "lightgbm")
        estimator = LGBMRegressor(**params)
    elif model_type == "quantile_lightgbm":
        from lightgbm import LGBMRegressor

        quantiles = model_def.quantiles or [0.5]
        quantile_models = {}
        for q in quantiles:
            q_params = {
                "objective": "quantile",
                "alpha": q,
                "n_estimators": params.get("n_estimators", 500),
                "n_jobs": params.get("n_jobs", -1),
            }
            q_params.update(params)
            q_params["objective"] = "quantile"
            q_params["alpha"] = q
            if model_def.monotone_constraints:
                q_params = _apply_monotone_constraints(q_params, model_def.monotone_constraints, "lightgbm")
            estimator = LGBMRegressor(**q_params)
            quantile_models[q] = estimator
        estimator = quantile_models
        extras["quantiles"] = quantiles
    elif model_type == "catboost":
        from catboost import CatBoostRegressor

        params.setdefault("loss_function", "RMSE")
        params.setdefault("verbose", 0)
        estimator = CatBoostRegressor(**params)
    elif model_type == "ngboost":
        from ngboost import NGBRegressor
        from ngboost.distns import LogNormal, Normal
        try:
            from ngboost.distns import Beta
        except ImportError:  # pragma: no cover
            Beta = None

        distribution = (model_def.distribution or "lognormal").lower()
        dist_map = {
            "lognormal": LogNormal,
            "normal": Normal,
        }
        if Beta is not None:
            dist_map["beta"] = Beta
        dist_cls = dist_map.get(distribution, LogNormal)
        params.setdefault("learning_rate", 0.01)
        estimator = NGBRegressor(Dist=dist_cls, **params)
    elif model_type == "stacked":
        base_defs = model_def.base_models or []
        base_estimators = []
        for idx, base_def in enumerate(base_defs):
            base_estimator, _ = _instantiate_estimator(base_def, feature_columns)
            base_name = base_def.model_type if base_def.model_type != "stacked" else f"model_{idx}"
            base_estimators.append((f"{base_name}_{idx}", base_estimator))
        if not base_estimators:
            raise ValueError("Stacked model requires at least one base_model definition.")
        meta_params = params.get("meta", {}) if isinstance(params.get("meta"), dict) else params
        meta_estimator = Ridge(**meta_params)
        estimator = StackingRegressor(estimators=base_estimators, final_estimator=meta_estimator)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return estimator, extras


def train_models(
    training_input: ModelTrainingInput,
    config: ModelTrainingConfig,
) -> ModelTrainingResults:
    features = training_input.features
    targets = training_input.targets
    results: Dict[str, TargetTrainingResult] = {}

    for target_config in config.targets:
        target_name = target_config.target
        if target_name not in targets.columns:
            logger.warning("Target '%s' not present in dataset; skipping", target_name)
            continue

        y = targets[target_name].dropna()
        if y.empty:
            logger.warning("Target '%s' has no valid samples; skipping", target_name)
            continue

        X = features.loc[y.index]
        model_results: List[ModelResult] = []

        for model_def in target_config.models:
            estimator, extras = _instantiate_estimator(model_def, list(features.columns))

            # Fit estimator
            if isinstance(estimator, dict):
                # Quantile LightGBM case
                quantile_predictions: Dict[float, np.ndarray] = {}
                for q, q_model in estimator.items():
                    q_model.fit(X, y)
                    quantile_predictions[q] = q_model.predict(X)
                median_quantile = extras.get("quantiles", [0.5])
                median_q = min(median_quantile, key=lambda val: abs(val - 0.5))
                y_pred = quantile_predictions[median_q]
                extras["quantile_predictions"] = {str(q): preds.tolist() for q, preds in quantile_predictions.items()}
                fitted_estimator = {q: q_model for q, q_model in estimator.items()}
            else:
                estimator.fit(X, y)
                y_pred = estimator.predict(X)
                fitted_estimator = estimator
                if model_def.model_type == "ngboost":
                    try:
                        extras["predicted_std"] = estimator.pred_dist(X).scale.tolist()
                    except AttributeError:  # pragma: no cover
                        pass

            metrics = _compute_metrics(y, y_pred)
            model_results.append(
                ModelResult(
                    name=model_def.model_type,
                    estimator=fitted_estimator,
                    metrics=metrics,
                    extras=extras,
                )
            )

            logger.info(
                "Trained %s for target '%s' (R²=%.3f, MAE=%.3f)",
                model_def.model_type,
                target_name,
                metrics.r2,
                metrics.mae,
            )

        results[target_name] = TargetTrainingResult(target=target_name, models=model_results)

    return ModelTrainingResults(targets=results)


@step
def model_training_step(
    training_input: ModelTrainingInput,
    split_output: SplitOutput,
    config: ModelTrainingConfig,
) -> ModelTrainingResults:
    """Train all configured models for each target and return metrics."""

    del split_output  # placeholder until evaluation incorporates test set
    training_results = train_models(training_input, config)
    logger.info("Completed training for targets: %s", list(training_results.targets.keys()))
    return training_results
