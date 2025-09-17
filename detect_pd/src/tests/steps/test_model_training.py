"""Tests for model training helper."""
from __future__ import annotations

import pandas as pd

from detect_pd.config import ModelDefinition, ModelTrainingConfig, TargetModelCollection
from detect_pd.steps.model_training import ModelTrainingResults, train_models
from detect_pd.steps.training_input import ModelTrainingInput
from detect_pd.steps.preprocessing import PreprocessingArtifacts


def build_training_input() -> ModelTrainingInput:
    features = pd.DataFrame({"signal": [0.0, 1.0, 2.0, 3.0], "noise": [1.0, 1.0, 1.0, 1.0]})
    targets = pd.DataFrame({"ktv": [0.0, 2.0, 4.0, 6.0]})
    artifacts = PreprocessingArtifacts(
        scaler=None,
        encoder=None,
        imputer=None,
        feature_columns=list(features.columns),
        encoded_categorical_columns=[],
        original_categorical_columns=[],
        numeric_columns=list(features.columns),
    )
    return ModelTrainingInput(
        features=features,
        targets=targets,
        selected_feature_map={"ktv": ["signal", "noise"]},
        artifacts=artifacts,
    )


def build_training_config() -> ModelTrainingConfig:
    elastic_net = ModelDefinition(model_type="elastic_net", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    xgboost = ModelDefinition(model_type="xgboost", hyperparameters={"n_estimators": 10, "max_depth": 3})
    target_collection = TargetModelCollection(target="ktv", models=[elastic_net, xgboost])
    return ModelTrainingConfig(targets=[target_collection], n_jobs=1, persist_models=False)


def test_train_models_returns_results():
    training_input = build_training_input()
    config = build_training_config()

    results: ModelTrainingResults = train_models(training_input, config)

    assert "ktv" in results.targets
    model_names = {model.name for model in results.targets["ktv"].models}
    assert {"elastic_net", "xgboost"}.issubset(model_names)
    for model_result in results.targets["ktv"].models:
        assert model_result.metrics.r2 >= 0.0
