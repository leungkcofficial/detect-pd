"""Configuration models for data ingestion and validation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import Field, field_validator

from .base import BaseConfig


class DataIngestionConfig(BaseConfig):
    """Settings controlling how the CRF Excel data is loaded and validated."""

    file_path: Path = Field(..., description="Path to the CRF Excel file.")
    sheet_name: str = Field("Sheet1", description="Excel sheet containing patient data.")
    header_rows: List[int] = Field(
        default_factory=lambda: [0, 1],
        description=(
            "Row indices used as header when reading the Excel file. "
            "Multiple rows support hierarchical column names (e.g., merged cells)."
        ),
    )
    required_columns: List[str] = Field(
        ..., description="Columns that must be present in the input dataset."
    )
    date_columns: List[str] = Field(
        default_factory=list,
        description="Columns that should contain parseable dates in chronological order.",
    )
    numeric_validation_rules: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Per-column numeric bounds expressed as (min, max).",
    )
    column_renames: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional mapping applied after ingestion to standardise column names. "
            "Useful when source data uses multi-level headers or inconsistent labels."
        ),
    )
    drop_columns: List[str] = Field(
        default_factory=list,
        description="Columns to discard after ingestion and renaming steps.",
    )
    index_column: Optional[str] = Field(
        None, description="Column to set as DataFrame index post ingestion (e.g., patient ID)."
    )
    drop_missing_outcomes: bool = Field(
        True,
        description="Drop rows where outcome variables are missing to enforce complete-case analysis.",
    )

    @field_validator("sheet_name")
    @classmethod
    def _strip_sheet_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("required_columns", "date_columns", "drop_columns", mode="before")
    @classmethod
    def _normalise_columns(cls, value: Optional[List[str]]) -> List[str]:
        if not value:
            return []
        return [column.strip() for column in value]

    @field_validator("column_renames", mode="before")
    @classmethod
    def _strip_rename_keys(cls, value: Optional[Dict[str, str]]) -> Dict[str, str]:
        if not value:
            return {}
        return {key.strip(): val.strip() for key, val in value.items()}


DataIngestionConfig.model_rebuild()
