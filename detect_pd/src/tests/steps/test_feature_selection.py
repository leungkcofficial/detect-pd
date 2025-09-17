"""Tests for LASSO-based feature selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from detect_pd.config import FeatureEngineeringConfig, FeatureSelectionTargetConfig
from detect_pd.steps.feature_selection import FeatureSelectionOutput, run_feature_selection
from detect_pd.steps.preprocessing import PreprocessingArtifacts, PreprocessingOutput


def build_preprocessing_output() -> PreprocessingOutput:
    rng = np.random.default_rng(42)
    n_samples = 120
    signal1 = rng.normal(0, 1, size=n_samples)
    signal2 = rng.normal(0, 1, size=n_samples)
    noise = rng.normal(0, 0.5, size=n_samples)

    features = pd.DataFrame(
        {
            "signal_one": signal1,
            "signal_two": signal2,
            "noise": rng.normal(0, 1, size=n_samples),
        }
    )
    target = 3 * signal1 - 2 * signal2 + noise
    targets = pd.DataFrame({"ktv": target}, index=features.index)

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


def build_config() -> FeatureEngineeringConfig:
    target_config = FeatureSelectionTargetConfig(
        target_name="ktv",
        problem_type="regression",
        alpha=1.0,
        max_iter=5000,
        min_features=2,
    )
    return FeatureEngineeringConfig(targets={"ktv": target_config})


def test_run_feature_selection_returns_expected_features(tmp_path: Path) -> None:
    preprocessing_output = build_preprocessing_output()
    config = build_config()

    result: FeatureSelectionOutput = run_feature_selection(
        preprocessing_output,
        config,
        output_dir=tmp_path,
    )

    assert "ktv" in result.results
    selection = result.results["ktv"]
    assert {"signal_one", "signal_two"}.issubset(selection.selected_features)
    assert selection.mse_trace_path.exists()
    assert selection.shap_summary_path.exists()
