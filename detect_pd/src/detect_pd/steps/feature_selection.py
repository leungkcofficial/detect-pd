"""Feature selection step using LASSO with interpretability plots."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from zenml import step

from detect_pd.config import FeatureEngineeringConfig, FeatureSelectionTargetConfig
from detect_pd.steps.preprocessing import PreprocessingOutput

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Stores the outcome of feature selection for a single target."""

    selected_features: List[str]
    coefficients: Dict[str, float]
    optimal_alpha: float
    mse_trace_path: Path
    shap_summary_path: Path


@dataclass
class FeatureSelectionOutput:
    """Aggregated feature selection results for all configured targets."""

    results: Dict[str, FeatureSelectionResult]


def _prepare_target_series(
    targets: pd.DataFrame,
    target_column: str,
    config: FeatureSelectionTargetConfig,
) -> pd.Series:
    if target_column not in targets.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")
    series = targets[target_column].dropna()
    if series.empty:
        raise ValueError(f"Target column '{target_column}' has no valid values.")

    if config.problem_type == "binary" and config.threshold is not None:
        series = (series >= config.threshold).astype(int)
    return series


def _fit_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    config: FeatureSelectionTargetConfig,
) -> LassoCV:
    alphas = np.logspace(-4, 2, num=80) * config.alpha
    model = LassoCV(alphas=alphas, cv=5, max_iter=config.max_iter, n_jobs=-1)
    model.fit(X, y)
    return model


def _select_features(
    features: pd.DataFrame,
    model: LassoCV,
    config: FeatureSelectionTargetConfig,
) -> List[str]:
    coef_series = pd.Series(model.coef_, index=features.columns)
    non_zero = coef_series[coef_series != 0.0]
    if len(non_zero) < config.min_features:
        non_zero = coef_series.abs().sort_values(ascending=False).head(config.min_features)
    return non_zero.index.tolist()


def _plot_lasso_path(model: LassoCV, output_path: Path) -> None:
    alphas = model.alphas_
    mse = model.mse_path_.mean(axis=1)
    plt.figure(figsize=(8, 5))
    plt.plot(np.log10(alphas), mse, marker="o")
    plt.axvline(np.log10(model.alpha_), color="red", linestyle="--", label=f"alpha*={model.alpha_:.4f}")
    plt.xlabel("log10(lambda)")
    plt.ylabel("Mean CV MSE")
    plt.title("LASSO Cross-Validation Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_shap_values(
    model: LassoCV,
    features: pd.DataFrame,
    selected_features: List[str],
    output_path: Path,
) -> None:
    if not selected_features:
        selected_features = features.columns.tolist()
    selected_frame = features[selected_features]
    explainer = shap.LinearExplainer(model, selected_frame)
    shap_values = explainer.shap_values(selected_frame)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        selected_frame,
        plot_type="bar",
        show=False,
        color="#1f77b4",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_feature_selection(
    preprocessing_output: PreprocessingOutput,
    config: FeatureEngineeringConfig,
    output_dir: Path | None = None,
) -> FeatureSelectionOutput:
    if output_dir is None:
        output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = preprocessing_output.features
    targets = preprocessing_output.targets
    scaler = StandardScaler(with_mean=False)
    scaled_features = pd.DataFrame(
        scaler.fit_transform(feature_frame),
        columns=feature_frame.columns,
        index=feature_frame.index,
    )

    results: Dict[str, FeatureSelectionResult] = {}
    for target_alias, target_cfg in config.targets.items():
        target_column = target_cfg.target_name or target_alias
        logger.info("Running feature selection for target '%s'", target_column)
        y_series = _prepare_target_series(targets, target_column, target_cfg)
        X_aligned = scaled_features.loc[y_series.index]

        model = _fit_lasso(X_aligned, y_series, target_cfg)
        selected = _select_features(X_aligned, model, target_cfg)

        mse_path = output_dir / f"lasso_path_{target_column}.png"
        shap_path = output_dir / f"shap_summary_{target_column}.png"
        _plot_lasso_path(model, mse_path)
        _plot_shap_values(model, X_aligned, selected, shap_path)

        coef_series = pd.Series(model.coef_, index=X_aligned.columns)
        coefficients = {feature: float(coef_series[feature]) for feature in selected}

        results[target_column] = FeatureSelectionResult(
            selected_features=selected,
            coefficients=coefficients,
            optimal_alpha=float(model.alpha_),
            mse_trace_path=mse_path,
            shap_summary_path=shap_path,
        )

        logger.info(
            "Target '%s': selected %d features (alpha=%.4f)",
            target_column,
            len(selected),
            model.alpha_,
        )

    return FeatureSelectionOutput(results=results)


@step
def feature_selection_step(
    preprocessing_output: PreprocessingOutput,
    config: FeatureEngineeringConfig,
) -> FeatureSelectionOutput:
    output_dir = Path("artifacts") / "feature_selection"
    output = run_feature_selection(preprocessing_output, config, output_dir=output_dir)

    summary_path = output_dir / "selected_features.json"
    summary = {
        target: {
            "selected_features": result.selected_features,
            "coefficients": result.coefficients,
            "alpha": result.optimal_alpha,
            "mse_plot": str(result.mse_trace_path),
            "shap_plot": str(result.shap_summary_path),
        }
        for target, result in output.results.items()
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Feature selection completed; summary saved to %s", summary_path)
    return output
