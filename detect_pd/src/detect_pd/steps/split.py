"""Dataset splitting step for the DETECT-PD pipeline."""

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step

from detect_pd.config import SplitConfig

logger = logging.getLogger(__name__)


@dataclass
class SplitOutput:
    """Output structure returned by the split step."""

    train: pd.DataFrame
    test: pd.DataFrame
    test_indices: List[str]


def perform_split(df: pd.DataFrame, config: SplitConfig) -> SplitOutput:
    """Split a DataFrame according to the configured ratios."""

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        shuffle=config.shuffle,
        random_state=config.random_seed,
    )

    test_indices = test_df.index.astype(str).tolist()
    logger.info(
        "Split dataset into %d training and %d test records (test indices: %s)",
        len(train_df),
        len(test_df),
        test_indices,
    )
    return SplitOutput(train=train_df, test=test_df, test_indices=test_indices)


@step
def split_step(data: pd.DataFrame, config: SplitConfig) -> SplitOutput:
    """ZenML step to partition the dataset into train/test subsets."""

    return perform_split(data, config)
