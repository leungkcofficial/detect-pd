"""Tests for the dataset split helper."""
from __future__ import annotations

import pandas as pd

from detect_pd.config import SplitConfig
from detect_pd.steps.split import SplitOutput, perform_split


def test_perform_split_is_deterministic() -> None:
    df = pd.DataFrame({"value": range(10)}, index=[f"patient-{i}" for i in range(10)])
    config = SplitConfig(test_size=0.2, random_seed=42)

    output_first = perform_split(df, config)
    output_second = perform_split(df, config)

    assert isinstance(output_first, SplitOutput)
    assert len(output_first.train) == 8
    assert len(output_first.test) == 2
    assert output_first.test_indices == output_second.test_indices


def test_perform_split_respects_shuffle_flag() -> None:
    df = pd.DataFrame({"value": range(5)})
    config_no_shuffle = SplitConfig(test_size=0.4, random_seed=1, shuffle=False)

    output = perform_split(df, config_no_shuffle)
    # Without shuffling, the tail of the dataset becomes the test split
    assert output.test_indices == [str(i) for i in range(3, 5)]
