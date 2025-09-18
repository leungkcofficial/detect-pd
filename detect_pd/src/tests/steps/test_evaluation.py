"""Tests for the evaluation helper."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from detect_pd.config import EvaluationConfig, PreprocessingConfig
from detect_pd.steps.evaluation import EvaluationSummary, evaluate_models
from detect_pd.steps.model_training import (
    ModelMetrics,
    ModelResult,
    ModelTrainingResults,
    TargetTrainingResult,
)
from detect_pd.steps.preprocessing import PreprocessingArtifacts
from detect_pd.steps.split import SplitOutput
from detect_pd.steps.training_input import ModelTrainingInput


def build_training_input() -> ModelTrainingInput:
    features = pd.DataFrame({"signal": [0.0, 1.0, 2.0, 3.0]})
    targets = pd.DataFrame({"ktv": [0.0, 2.0, 4.0, 6.0]})
    artifacts = PreprocessingArtifacts(
        scaler=None,
        encoder=None,
        imputer=None,
        feature_columns=["signal"],
        encoded_categorical_columns=[],
        original_categorical_columns=[],
        numeric_columns=["signal"],
    )
    return ModelTrainingInput(
        features=features,
        targets=targets,
        selected_feature_map={"ktv": ["signal"]},
        artifacts=artifacts,
    )


def build_training_results() -> ModelTrainingResults:
    X = pd.DataFrame({"signal": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 2.0, 4.0, 6.0])
    estimator = LinearRegression().fit(X, y)
    metrics = ModelMetrics(r2=1.0, mae=0.0, mse=0.0)
    model_result = ModelResult(
        name="linear_regression",
        estimator=estimator,
        metrics=metrics,
        extras={"best_params": estimator.get_params(), "best_score": 1.0},
    )
    target_result = TargetTrainingResult(target="ktv", models=[model_result])
    return ModelTrainingResults(targets={"ktv": target_result})


def build_split_output() -> SplitOutput:
    train = pd.DataFrame({"signal": [0.0, 1.0], "ktv": [0.0, 2.0]})
    test = pd.DataFrame({"signal": [4.0, 5.0], "ktv": [8.0, 10.0]})
    return SplitOutput(train=train, test=test, test_indices=["A", "B"])


def test_evaluate_models_produces_metrics(tmp_path: Path):
    training_input = build_training_input()
    training_results = build_training_results()
    split_output = build_split_output()

    preprocessing_config = PreprocessingConfig(
        target_columns=["ktv"],
        numeric_features=["signal"],
    )
    eval_config = EvaluationConfig(generate_plots=False, output_dir=str(tmp_path))

    summary: EvaluationSummary = evaluate_models(
        training_results=training_results,
        training_input=training_input,
        split_output=split_output,
        preprocessing_config=preprocessing_config,
        config=eval_config,
    )

    assert "ktv" in summary.targets
    eval_result = summary.targets["ktv"].models[0]
    assert eval_result.metrics.r2 >= 0.0
