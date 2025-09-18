"""Preprocessing step for DETECT-PD pipeline."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from zenml import step

from detect_pd.config import PreprocessingConfig
from detect_pd.utils import (
    CCI_BASE_WEIGHTS,
    CharlsonConfig,
    compute_bmi,
    compute_bsa_du_bois,
    compute_charlson_index,
    derive_time_features,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingArtifacts:
    """Artifacts produced by the preprocessing stage."""

    scaler: Optional[Any]
    encoder: Optional[Any]
    imputer: Optional[Any]
    feature_columns: List[str]
    encoded_categorical_columns: List[str]
    original_categorical_columns: List[str]
    numeric_columns: List[str]


@dataclass
class PreprocessingOutput:
    """Structured output returned by the preprocessing step."""

    features: pd.DataFrame
    targets: pd.DataFrame
    artifacts: PreprocessingArtifacts


def _ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _apply_log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            values = df[column]
            if (values < -1).any():
                raise ValueError(
                    f"Cannot apply log1p to column '{column}' with values < -1."
                )
            df[column] = np.log1p(values)
    return df


def _compute_cci(df: pd.DataFrame, config: PreprocessingConfig) -> pd.Series:
    if not config.comorbidity_columns:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index, name="charlson_index")

    weights = dict(CCI_BASE_WEIGHTS)
    if config.cci_weights:
        weights.update(config.cci_weights)
    charlson_config = CharlsonConfig(
        weights=weights,
        include_age=config.include_age_in_cci,
    )

    age_column = config.age_column

    def row_score(row: pd.Series) -> int:
        comorbidities = [
            charlson_label
            for column, charlson_label in config.comorbidity_columns.items()
            if column in row and bool(row[column])
        ]
        age_value = row.get(age_column) if age_column in row else None
        return compute_charlson_index(comorbidities, age=age_value, config=charlson_config)

    return df.apply(row_score, axis=1).rename("charlson_index")


def _derive_body_metrics(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    weight_col = config.weight_column
    height_col = config.height_column
    if weight_col and height_col and weight_col in df.columns and height_col in df.columns:
        weights = _ensure_numeric(df[weight_col])
        heights = _ensure_numeric(df[height_col])
        bmi_values: List[float] = []
        bsa_values: List[float] = []
        for weight, height in zip(weights, heights):
            if pd.isna(weight) or pd.isna(height):
                bmi_values.append(np.nan)
                bsa_values.append(np.nan)
                continue
            try:
                bmi_values.append(compute_bmi(float(weight), float(height)))
                bsa_values.append(compute_bsa_du_bois(float(weight), float(height)))
            except ValueError:
                bmi_values.append(np.nan)
                bsa_values.append(np.nan)
        df["bmi"] = bmi_values
        df["bsa"] = bsa_values
    return df


def _derive_time_metrics(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    if not config.time_column_map:
        return df
    missing_columns = [source for source in config.time_column_map.values() if source not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns for time feature derivation: {missing_columns}")
    time_features = derive_time_features(df, config.time_column_map)
    return pd.concat([df, time_features], axis=1)


def _split_features_targets(df: pd.DataFrame, target_columns: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = df[target_columns].copy() if target_columns else pd.DataFrame(index=df.index)
    features = df.drop(columns=[col for col in target_columns if col in df.columns])
    return features, targets


def _impute_numeric_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    strategy: str,
) -> tuple[pd.DataFrame, Optional[SimpleImputer]]:
    if strategy == "none" or not numeric_columns:
        return df, None

    imputer = SimpleImputer(strategy=strategy)
    numeric_values = df[numeric_columns]
    df.loc[:, numeric_columns] = imputer.fit_transform(numeric_values)
    return df, imputer


def _scale_numeric_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, Optional[Any]]:
    if not numeric_columns:
        return df, None

    scaler: Any
    if config.scaling_method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=config.minmax_feature_range)

    numerical_values = df[numeric_columns].astype(float)
    scaled_values = scaler.fit_transform(numerical_values)
    df.loc[:, numeric_columns] = scaled_values
    return df, scaler


def _encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    encoding: str,
) -> tuple[pd.DataFrame, Optional[Any], List[str]]:
    if not categorical_columns:
        return df, None, categorical_columns

    if encoding == "one_hot":
        encoder_args: Dict[str, Any] = {"handle_unknown": "ignore"}
        if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
            encoder_args["sparse_output"] = False
        else:
            encoder_args["sparse"] = False
        encoder = OneHotEncoder(**encoder_args)
        transformed = encoder.fit_transform(df[categorical_columns].fillna("missing"))
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(transformed, index=df.index, columns=encoded_columns)
        df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
        return df, encoder, list(encoded_columns)

    # Label encoding fallback (applied column-wise)
    from sklearn.preprocessing import LabelEncoder

    encoder_map: Dict[str, LabelEncoder] = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str).fillna("missing"))
        encoder_map[column] = le
    return df, encoder_map, categorical_columns


def preprocess_dataset(data: pd.DataFrame, config: PreprocessingConfig) -> PreprocessingOutput:
    df = data.copy()

    df = _derive_body_metrics(df, config)
    df = _derive_time_metrics(df, config)
    df["charlson_index"] = _compute_cci(df, config)

    df = _apply_log_transform(df, config.log_transform_features)

    features, targets = _split_features_targets(df, config.target_columns)

    inferred_numeric = config.numeric_features or [
        column
        for column in features.select_dtypes(include=[np.number]).columns
    ]
    categorical_columns = config.categorical_features

    features, imputer = _impute_numeric_features(features, inferred_numeric, config.imputation_strategy)
    features, scaler = _scale_numeric_features(features, inferred_numeric, config)
    features, encoder, encoded_columns = _encode_categorical_features(
        features, categorical_columns, config.categorical_encoding
    )

    artifacts = PreprocessingArtifacts(
        scaler=scaler,
        encoder=encoder,
        imputer=imputer,
        feature_columns=list(features.columns),
        encoded_categorical_columns=encoded_columns,
        original_categorical_columns=categorical_columns,
        numeric_columns=inferred_numeric,
    )
    return PreprocessingOutput(features=features, targets=targets, artifacts=artifacts)


@step
def preprocessing_step(
    data: pd.DataFrame,
    config: PreprocessingConfig,
) -> PreprocessingOutput:
    """ZenML preprocessing step that prepares features and artifacts."""

    output = preprocess_dataset(data, config)
    logger.info("Preprocessed dataset with %d features", output.features.shape[1])
    return output


def apply_preprocessing_to_new_data(
    data: pd.DataFrame,
    config: PreprocessingConfig,
    artifacts: PreprocessingArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fitted preprocessing artifacts to new data."""

    df = data.copy()
    df = _derive_body_metrics(df, config)
    df = _derive_time_metrics(df, config)
    df["charlson_index"] = _compute_cci(df, config)
    df = _apply_log_transform(df, config.log_transform_features)

    features, targets = _split_features_targets(df, config.target_columns)

    # Ensure numeric columns exist and apply imputer/scaler
    for column in artifacts.numeric_columns:
        if column not in features.columns:
            features[column] = np.nan

    numeric_columns = [col for col in artifacts.numeric_columns if col in features.columns]
    if artifacts.imputer is not None and numeric_columns:
        features.loc[:, numeric_columns] = artifacts.imputer.transform(features[numeric_columns])
    if artifacts.scaler is not None and numeric_columns:
        features.loc[:, numeric_columns] = artifacts.scaler.transform(features[numeric_columns])

    categorical_columns = [col for col in artifacts.original_categorical_columns if col in features.columns]
    if artifacts.encoder is not None and categorical_columns:
        if hasattr(artifacts.encoder, "transform"):
            transformed = artifacts.encoder.transform(features[categorical_columns].fillna("missing"))
            encoded_df = pd.DataFrame(
                transformed,
                index=features.index,
                columns=artifacts.encoded_categorical_columns,
            )
            features = pd.concat([features.drop(columns=categorical_columns), encoded_df], axis=1)
        elif isinstance(artifacts.encoder, dict):
            for column, encoder in artifacts.encoder.items():
                if column in features.columns:
                    features[column] = encoder.transform(features[column].astype(str).fillna("missing"))
    else:
        features = features.drop(columns=categorical_columns, errors="ignore")

    for column in artifacts.feature_columns:
        if column not in features.columns:
            features[column] = 0.0

    features = features[artifacts.feature_columns]
    return features, targets
