"""Data ingestion step for the DETECT-PD pipeline."""

import logging
from typing import Iterable

import pandas as pd
from zenml import step

from detect_pd.config import DataIngestionConfig
from detect_pd.utils import (
    flatten_multiindex_columns,
    parse_dates,
    rename_and_select_columns,
)

logger = logging.getLogger(__name__)


def _validate_numeric_ranges(df: pd.DataFrame, rules: dict[str, tuple[float, float]]) -> None:
    """Ensure numeric columns fall within configured bounds."""

    for column, (minimum, maximum) in rules.items():
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = (~series.between(minimum, maximum)) & series.notna()
        if invalid_mask.any():
            invalid_values = series[invalid_mask].tolist()
            raise ValueError(
                f"Column '{column}' contains values outside range [{minimum}, {maximum}]: {invalid_values}"
            )


def _drop_rows_with_missing(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    subset = [col for col in columns if col in df.columns]
    if not subset:
        return df
    before = len(df)
    df = df.dropna(subset=subset)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %s rows due to missing values in %s", dropped, subset)
    return df


def ingest_data_frame(config: DataIngestionConfig) -> pd.DataFrame:
    """Load, validate, and clean the DETECT-PD CRF dataset."""

    header = config.header_rows or 0
    df = pd.read_excel(
        config.file_path,
        sheet_name=config.sheet_name,
        header=header,
        engine=None,
    )

    df = flatten_multiindex_columns(df)
    df = rename_and_select_columns(
        df,
        renames=config.column_renames,
        drop_columns=config.drop_columns,
        required_columns=config.required_columns,
    )
    df = parse_dates(df, config.date_columns)

    _validate_numeric_ranges(df, config.numeric_validation_rules)

    if config.drop_missing_outcomes:
        df = _drop_rows_with_missing(df, config.required_columns)

    if config.index_column and config.index_column in df.columns:
        df = df.set_index(config.index_column, drop=True)

    logger.info("Ingested %d patient records", len(df))
    return df


@step
def data_ingestion_step(config: DataIngestionConfig) -> pd.DataFrame:
    """ZenML step wrapper for data ingestion."""

    return ingest_data_frame(config)
