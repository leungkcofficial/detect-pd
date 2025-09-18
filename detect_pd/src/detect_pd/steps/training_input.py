"""Utilities to prepare model training inputs from selected features."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd
from zenml import step

if TYPE_CHECKING:
    from detect_pd.steps.feature_selection import FeatureSelectionOutput
    from detect_pd.steps.preprocessing import PreprocessingArtifacts, PreprocessingOutput
else:  # pragma: no cover
    FeatureSelectionOutput = Any
    PreprocessingArtifacts = Any
    PreprocessingOutput = Any

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingInput:
    """Payload passed to the model training step."""

    features: pd.DataFrame
    targets: pd.DataFrame
    selected_feature_map: Dict[str, List[str]]
    artifacts: PreprocessingArtifacts


def build_training_input(
    preprocessing_output: "PreprocessingOutput",
    feature_selection_output: "FeatureSelectionOutput",
) -> ModelTrainingInput:
    selected_feature_map = {
        target: result.selected_features for target, result in feature_selection_output.results.items()
    }
    if selected_feature_map:
        selected_union = sorted({feature for features in selected_feature_map.values() for feature in features})
        missing_features = [feature for feature in selected_union if feature not in preprocessing_output.features.columns]
        if missing_features:
            logger.warning("Some selected features were not found in the feature matrix: %s", missing_features)
        filtered_features = preprocessing_output.features.loc[:, [f for f in selected_union if f in preprocessing_output.features.columns]].copy()
    else:
        logger.warning("No features were selected; falling back to entire feature matrix")
        filtered_features = preprocessing_output.features.copy()

    return ModelTrainingInput(
        features=filtered_features,
        targets=preprocessing_output.targets.copy(),
        selected_feature_map=selected_feature_map,
        artifacts=preprocessing_output.artifacts,
    )


try:
    @step
    def prepare_training_input_step(
        preprocessing_output: "PreprocessingOutput",
        feature_selection_output: "FeatureSelectionOutput",
    ) -> ModelTrainingInput:
        """ZenML step wrapper around :func:`build_training_input`."""

        return build_training_input(preprocessing_output, feature_selection_output)

except RuntimeError:

    def prepare_training_input_step(
        preprocessing_output: "PreprocessingOutput",
        feature_selection_output: "FeatureSelectionOutput",
    ) -> ModelTrainingInput:  # type: ignore[misc]
        logger.warning("ZenML step decoration disabled; executing pure helper function instead.")
        return build_training_input(preprocessing_output, feature_selection_output)
