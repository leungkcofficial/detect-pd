"""ZenML pipeline entry point for DETECT-PD."""

import argparse
from pathlib import Path
from typing import Optional

import yaml
from zenml import pipeline

from detect_pd.config.pipeline import PipelineConfig
from detect_pd.steps import (
    EvaluationSummary,
    ModelTrainingInput,
    ModelTrainingResults,
    SplitOutput,
    data_ingestion_step,
    evaluation_step,
    feature_selection_step,
    model_training_step,
    prepare_training_input_step,
    preprocessing_step,
    split_step,
)


@pipeline
def training_pipeline(config: PipelineConfig) -> EvaluationSummary:
    """Full pipeline that trains models and evaluates them on the test set."""

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
    training_results = model_training_step(
        training_input=training_input,
        split_output=split_output,
        config=config.model_training,
    )

    evaluation_summary = evaluation_step(
        training_results=training_results,
        training_input=training_input,
        split_output=split_output,
        preprocessing_config=config.preprocessing,
        config=config.evaluation,
    )

    return evaluation_summary


def load_pipeline_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return PipelineConfig.parse_obj(data)


def run_pipeline(config: PipelineConfig) -> None:
    pipeline_instance = training_pipeline(config=config)
    pipeline_instance.run()


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DETECT-PD training pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the pipeline YAML/JSON configuration file.",
    )
    return parser.parse_args(args=args)


def main(argv: Optional[list[str]] = None) -> None:
    parsed_args = parse_args(argv)
    config_path = Path(parsed_args.config)
    config = load_pipeline_config(config_path)
    run_pipeline(config)


if __name__ == "__main__":
    main()
