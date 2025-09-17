"""Placeholder model training step that will be extended in future tasks."""
from __future__ import annotations

import logging

from zenml import step

from detect_pd.steps.split import SplitOutput
from detect_pd.steps.training_input import ModelTrainingInput

logger = logging.getLogger(__name__)


@step
def model_training_step(training_input: ModelTrainingInput, split_output: SplitOutput) -> None:
    """Placeholder training step receiving selected features for future modelling."""

    logger.info(
        "Received training dataset with %d samples and %d selected features",
        len(training_input.features),
        training_input.features.shape[1],
    )
    logger.info("Targets available: %s", list(training_input.targets.columns))
    logger.info("Selected feature map: %s", training_input.selected_feature_map)
    # Actual model training logic will be implemented in a subsequent task.
