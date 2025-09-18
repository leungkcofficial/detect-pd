"""Configuration for dataset splitting."""

from pydantic import Field, field_validator

from .base import BaseConfig


class SplitConfig(BaseConfig):
    """Controls how the dataset is partitioned into train/test subsets."""

    test_size: float = Field(0.2, ge=0.05, le=0.5, description="Proportion of data for the test split.")
    shuffle: bool = Field(True, description="Whether to shuffle before splitting.")

    @field_validator("test_size")
    @classmethod
    def _validate_test_size(cls, value: float) -> float:
        if not (0.0 < value < 1.0):
            raise ValueError("test_size must be between 0 and 1.")
        return value


SplitConfig.model_rebuild()
