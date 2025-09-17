"""ZenML pipeline wiring data ingestion through feature selection."""
from __future__ import annotations

from zenml import pipeline

from detect_pd.config import (
    DataIngestionConfig,
    FeatureEngineeringConfig,
    PipelineConfig,
    PreprocessingConfig,
    SplitConfig,
)
from detect_pd.steps import (
    ModelTrainingInput,
    SplitOutput,
    data_ingestion_step,
    feature_selection_step,
    prepare_training_input_step,
    preprocessing_step,
    split_step,
)
from detect_pd.steps.model_training import model_training_step


@pipeline
def training_pipeline(config: PipelineConfig) -> ModelTrainingInput:
    """Full pipeline that prepares model training inputs."""

    # Ingestion
    raw_data = data_ingestion_step(config=config.data_ingestion)

    # Split into train/test
    split_output: SplitOutput = split_step(data=raw_data, config=config.split)

    # Preprocess training data only for downstream steps
    preprocessing_output = preprocessing_step(data=split_output.train, config=config.preprocessing)

    # Feature selection on preprocessed training set
    feature_selection_output = feature_selection_step(
        preprocessing_output=preprocessing_output,
        config=config.feature_engineering,
    )

    # Prepare dataset for training using selected features
    training_input = prepare_training_input_step(
        preprocessing_output=preprocessing_output,
        feature_selection_output=feature_selection_output,
    )

    # Model training stage (currently delegated to dedicated step module)
    model_training_step(training_input=training_input, split_output=split_output)

    return training_input
