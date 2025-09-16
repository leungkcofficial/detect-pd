"""Configuration around experiment tracking and artifact storage."""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from .base import BaseConfig


class TrackingConfig(BaseConfig):
    """Defines MLflow tracking properties and storage locations."""

    tracking_uri: Optional[str] = Field(
        None, description="MLflow tracking URI; defaults to local file store when None."
    )
    experiment_name: str = Field(
        "DETECT_PD_Pipeline", description="Name of the MLflow experiment to log runs under."
    )
    run_name_template: str = Field(
        "{pipeline_name}_{timestamp}",
        description="Template used to build MLflow run names.",
    )
    registry_uri: Optional[str] = Field(
        None, description="Optional MLflow model registry URI."
    )
