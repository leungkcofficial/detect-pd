"""Tests for preparing model training inputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from detect_pd.steps.preprocessing import PreprocessingArtifacts, PreprocessingOutput
from detect_pd.steps.training_input import ModelTrainingInput, build_training_input


def build_preprocessing_output() -> PreprocessingOutput:
    features = pd.DataFrame({"a": [1.0, 2.0], "b": [0.5, -0.5], "c": [3.0, 4.0]})
    targets = pd.DataFrame({"ktv": [1.1, 1.3], "pet": [0.8, 0.9]})
    artifacts = PreprocessingArtifacts(
        scaler=None,
        encoder=None,
        imputer=None,
        feature_columns=list(features.columns),
        encoded_categorical_columns=[],
        original_categorical_columns=[],
        numeric_columns=list(features.columns),
    )
    return PreprocessingOutput(features=features, targets=targets, artifacts=artifacts)


class DummyResult:
    def __init__(self, selected_features):
        self.selected_features = selected_features
        self.coefficients = {feature: 0.0 for feature in selected_features}
        self.optimal_alpha = 0.01
        self.mse_trace_path = Path("lasso.png")
        self.shap_summary_path = Path("shap.png")


class DummyFeatureSelectionOutput:
    def __init__(self, results):
        self.results = results


def build_feature_selection_output() -> DummyFeatureSelectionOutput:
    results = {
        "ktv": DummyResult(["a", "b"]),
        "pet": DummyResult(["b"]),
    }
    return DummyFeatureSelectionOutput(results=results)


def test_prepare_training_input_step_filters_features(monkeypatch):
    preprocessing_output = build_preprocessing_output()
    feature_selection_output = build_feature_selection_output()

    training_input: ModelTrainingInput = build_training_input(
        preprocessing_output=preprocessing_output,
        feature_selection_output=feature_selection_output,
    )

    assert list(training_input.features.columns) == ["a", "b"]
    assert training_input.targets.equals(preprocessing_output.targets)
    assert training_input.selected_feature_map["ktv"] == ["a", "b"]
    assert training_input.artifacts.feature_columns == preprocessing_output.artifacts.feature_columns
