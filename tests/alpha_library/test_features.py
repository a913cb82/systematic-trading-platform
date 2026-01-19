from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.alpha_library.features import residual_vol_20, returns_residual
from src.core.alpha_engine import FEATURES
from src.core.data_platform import Bar, DataPlatform
from src.core.types import QueryConfig, Timeframe

# Constants to avoid magic values
NUM_ASSETS = 5
RESIDUAL_LENGTH = 5
TOLERANCE = 1e-10


def test_feature_calculates_returns_residual_with_sufficient_data() -> None:
    assert returns_residual(pd.DataFrame(), Timeframe.MIN_30).empty
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now()] * 2,
            "internal_id": [1, 2],
            "returns_raw_30min": [0.01, 0.02],
        }
    )
    assert returns_residual(df, Timeframe.MIN_30).isna().all()


def test_feature_calculates_volatility_of_residuals() -> None:
    df = pd.DataFrame(
        {
            "internal_id": [1] * RESIDUAL_LENGTH,
            "returns_residual_30min": [0.1, 0.2, 0.1, 0.2, 0.1],
        }
    )
    assert len(residual_vol_20(df, Timeframe.MIN_30)) == RESIDUAL_LENGTH


def test_feature_calculation_robustness_across_random_returns(
    data_platform: DataPlatform,
) -> None:
    ts = datetime(2025, 1, 1, 12, 0)
    iids = [data_platform.get_internal_id(f"T{i}") for i in range(NUM_ASSETS)]
    tf = Timeframe.MIN_30
    all_bars = []
    for iid in iids:
        for i in range(10):
            price = 100 + np.random.randn()
            bar = Bar(
                iid,
                ts + timedelta(days=i),
                price,
                price + 1,
                price - 1,
                price,
                1000,
                timeframe=tf,
            )
            all_bars.append(bar)
    data_platform.add_bars(all_bars)

    query = QueryConfig(start=ts, end=ts + timedelta(days=9), timeframe=tf)
    df = data_platform.get_bars(iids, query)
    df["returns_raw_30min"] = FEATURES["returns_raw_30min"](df)
    df["returns_residual_30min"] = FEATURES["returns_residual_30min"](df)

    clean_df = df.dropna(subset=["returns_raw_30min"])
    res_values = clean_df["returns_residual_30min"].values
    assert np.any(np.abs(res_values) > TOLERANCE)
