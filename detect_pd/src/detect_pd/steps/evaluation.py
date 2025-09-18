"""Model evaluation step producing metrics and comparison plots."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from zenml import step

from detect_pd.config import EvaluationConfig, PreprocessingConfig
from detect_pd.steps.model_training import ModelTrainingResults
from detect_pd.steps.preprocessing import apply_preprocessing_to_new_data
from detect_pd.steps.split import SplitOutput
from detect_pd.steps.training_input import ModelTrainingInput

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    r2: float
    mae: float
    mse: float


@dataclass
class ModelEvaluation:
    name: str
    metrics: EvaluationMetrics
    extras: Dict[str, List[float]]


@dataclass
class TargetEvaluation:
    target: str
    models: List[ModelEvaluation]


@dataclass
class EvaluationSummary:
    targets: Dict[str, TargetEvaluation]
    discrimination_plots: Dict[str, Path]
    calibration_plots: Dict[str, Path]


def _plot_discrimination(
    target: str,
    evaluations: List[ModelEvaluation],
    metric_name: str,
    output_path: Path,
) -> None:
    names = [evaluation.name for evaluation in evaluations]
    values = [getattr(evaluation.metrics, metric_name) for evaluation in evaluations]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(names, values, color="#1f77b4")
    plt.title(f"{target.upper()} - {metric_name.upper()} Comparison")
    plt.ylabel(metric_name.upper())
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_calibration(
    target: str,
    evaluations: List[ModelEvaluation],
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    for evaluation in evaluations:
        y_true = np.array(evaluation.extras["y_true_test"])
        y_pred = np.array(evaluation.extras["y_pred_test"])
        plt.scatter(y_pred, y_true, s=18, alpha=0.6, label=evaluation.name)
    all_true = np.concatenate([np.array(evaluation.extras["y_true_test"]) for evaluation in evaluations])
    lims = [all_true.min(), all_true.max()]
    plt.plot(lims, lims, color="black", linestyle="--", label="Ideal")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title(f"Calibration Plot - {target.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def evaluate_models(
    training_results: ModelTrainingResults,
    training_input: ModelTrainingInput,
    split_output: SplitOutput,
    preprocessing_config: PreprocessingConfig,
    config: EvaluationConfig,
) -> EvaluationSummary:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = split_output.test.copy()
    processed_features, processed_targets = apply_preprocessing_to_new_data(
        test_data,
        preprocessing_config,
        training_input.artifacts,
    )

    targets_summary: Dict[str, TargetEvaluation] = {}
    discrimination_plots: Dict[str, Path] = {}
    calibration_plots: Dict[str, Path] = {}

    for target_name, target_result in training_results.targets.items():
        if target_name not in processed_targets.columns:
            logger.warning("Target '%s' missing from test set; skipping.", target_name)
            continue

        y_true = processed_targets[target_name].dropna()
        if y_true.empty:
            logger.warning("Target '%s' has no observations in test set; skipping.", target_name)
            continue

        feature_subset = training_input.selected_feature_map.get(
            target_name, list(processed_features.columns)
        )
        X_test = processed_features.loc[y_true.index, feature_subset]
        model_evaluations: List[ModelEvaluation] = []

        for model_result in target_result.models:
            estimator = model_result.estimator
            extras = dict(model_result.extras)

            if isinstance(estimator, dict):
                predictions = {}
                for quantile, quantile_estimator in estimator.items():
                    predictions[str(quantile)] = quantile_estimator.predict(X_test)
                median_quantile = min((float(q) for q in predictions.keys()), key=lambda val: abs(val - 0.5))
                y_pred = predictions[str(median_quantile)]
                extras["quantile_predictions_test"] = {
                    q: preds.tolist() for q, preds in predictions.items()
                }
            else:
                y_pred = estimator.predict(X_test)
                if model_result.name == "ngboost" and hasattr(estimator, "pred_dist"):
                    try:
                        extras["predicted_std_test"] = estimator.pred_dist(X_test).scale.tolist()
                    except AttributeError:  # pragma: no cover
                        pass

            metrics = EvaluationMetrics(
                r2=r2_score(y_true, y_pred),
                mae=mean_absolute_error(y_true, y_pred),
                mse=mean_squared_error(y_true, y_pred),
            )
            extras["y_true_test"] = y_true.tolist()
            extras["y_pred_test"] = y_pred.tolist()
            model_evaluations.append(
                ModelEvaluation(
                    name=model_result.name,
                    metrics=metrics,
                    extras=extras,
                )
            )

        if not model_evaluations:
            continue

        targets_summary[target_name] = TargetEvaluation(target=target_name, models=model_evaluations)

        if config.generate_plots:
            discrim_path = output_dir / f"discrimination_{target_name}.png"
            _plot_discrimination(target_name, model_evaluations, config.comparison_metric, discrim_path)
            discrimination_plots[target_name] = discrim_path

            calib_path = output_dir / f"calibration_{target_name}.png"
            _plot_calibration(target_name, model_evaluations, calib_path)
            calibration_plots[target_name] = calib_path

    return EvaluationSummary(
        targets=targets_summary,
        discrimination_plots=discrimination_plots,
        calibration_plots=calibration_plots,
    )


@step
def evaluation_step(
    training_results: ModelTrainingResults,
    training_input: ModelTrainingInput,
    split_output: SplitOutput,
    preprocessing_config: PreprocessingConfig,
    config: EvaluationConfig,
) -> EvaluationSummary:
    """Evaluate tuned models on the test split and produce comparison plots."""

    summary = evaluate_models(
        training_results=training_results,
        training_input=training_input,
        split_output=split_output,
        preprocessing_config=preprocessing_config,
        config=config,
    )
    logger.info("Evaluation completed for targets: %s", list(summary.targets.keys()))
    return summary
