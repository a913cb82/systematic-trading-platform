from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.alpha_library.models import (
    EarningsModel,
    MomentumModel,
    ReversionModel,
)
from src.core.alpha_engine import AlphaEngine, ModelRunConfig
from src.core.data_platform import Bar, DataPlatform, Event
from src.core.types import Timeframe

# Constants to avoid magic values
ASSET_1 = 1001
ASSET_2 = 1002
ASSET_3 = 1003
ASSET_4 = 1004
POSITIVE_SURPRISE = 0.05
NEGATIVE_SURPRISE = -0.05


def test_momentum_model_generates_signals_from_idiosyncratic_outperformance(
    populated_platform: Any,
) -> None:
    data, iid, ts = populated_platform
    all_iids = data.get_universe(ts)
    model = MomentumModel()
    config = ModelRunConfig(timestamp=ts, timeframe=Timeframe.MIN_30)
    signals = AlphaEngine.run_model(data, model, all_iids, config)
    assert iid in signals


def test_reversion_model_forecasts_mean_reversion_from_residual_returns() -> (
    None
):
    model = ReversionModel()
    df = pd.DataFrame(
        {
            "returns_residual_30min": [0.1, -0.2],
            "residual_vol_20_30min": [0.01, 0.02],
        },
        index=[ASSET_1, ASSET_2],
    )
    signals = model.compute_signals(df)
    assert signals[ASSET_1] == -(0.1 / 0.01)
    assert signals[ASSET_2] == -(-0.2 / 0.02)


def test_reversion_model_gracefully_handles_zero_volatility_and_nans() -> None:
    model = ReversionModel()
    df_edge = pd.DataFrame(
        {
            "returns_residual_30min": [0.1, np.nan],
            "residual_vol_20_30min": [0.0, 0.02],
        },
        index=[ASSET_3, ASSET_4],
    )
    signals_edge = model.compute_signals(df_edge)
    assert ASSET_3 not in signals_edge
    assert ASSET_4 not in signals_edge


def test_earnings_model_signals_positive_surprise_events(
    data_platform: DataPlatform,
) -> None:
    ts = datetime(2025, 1, 1, 12, 0)
    iid = data_platform.register_security("AAPL")
    bar = Bar(iid, ts, 100, 101, 99, 100, 1000, timeframe=Timeframe.DAY)
    data_platform.add_bars([bar])
    model = EarningsModel()
    config = ModelRunConfig(timestamp=ts, timeframe=Timeframe.DAY)

    res = AlphaEngine.run_model(data_platform, model, [iid], config)
    assert res.get(iid, 0.0) == 0.0

    event_ts = ts - timedelta(hours=2)
    ev = Event(
        iid,
        event_ts,
        "EARNINGS_RELEASE",
        {"surprise_pct": POSITIVE_SURPRISE},
        timestamp_knowledge=event_ts,
    )
    data_platform.add_events([ev])
    signals = AlphaEngine.run_model(data_platform, model, [iid], config)
    assert signals[iid] > 0


def test_earnings_model_ignores_negative_surprise_events(
    data_platform: DataPlatform,
) -> None:
    ts = datetime(2025, 1, 1, 12, 0)
    iid = data_platform.register_security("MSFT")
    bar = Bar(iid, ts, 100, 100, 100, 100, 1000, timeframe=Timeframe.DAY)
    data_platform.add_bars([bar])
    event_ts = ts - timedelta(hours=2)
    ev = Event(
        iid,
        event_ts,
        "EARNINGS_RELEASE",
        {"surprise_pct": NEGATIVE_SURPRISE},
        timestamp_knowledge=event_ts,
    )
    data_platform.add_events([ev])
    model = EarningsModel()
    config = ModelRunConfig(timestamp=ts, timeframe=Timeframe.DAY)
    signals = AlphaEngine.run_model(data_platform, model, [iid], config)
    assert signals[iid] == 0.0


def test_earnings_model_returns_zero_without_execution_context() -> None:
    model = EarningsModel()
    idx = [ASSET_1]
    df = pd.DataFrame(index=idx)
    assert model.compute_signals(df) == {ASSET_1: 0.0}


def test_earnings_signal_decays_linearly_over_time(
    data_platform: DataPlatform,
) -> None:
    ts = datetime(2025, 1, 1, 12, 0)
    iid = data_platform.register_security("AAPL")
    event_ts = ts - timedelta(hours=1)
    ev = Event(
        iid,
        event_ts,
        "EARNINGS_RELEASE",
        {"surprise_pct": POSITIVE_SURPRISE},
        timestamp_knowledge=event_ts,
    )
    data_platform.add_events([ev])
    bar1 = Bar(iid, ts, 100, 100, 100, 100, 1000, timeframe=Timeframe.DAY)
    data_platform.add_bars([bar1])
    model = EarningsModel()

    cfg1 = ModelRunConfig(timestamp=ts, timeframe=Timeframe.DAY)
    sig1 = AlphaEngine.run_model(data_platform, model, [iid], cfg1)[iid]

    future_ts = ts + timedelta(hours=10)
    bar2 = Bar(
        iid, future_ts, 100, 100, 100, 100, 1000, timeframe=Timeframe.DAY
    )
    data_platform.add_bars([bar2])
    cfg2 = ModelRunConfig(timestamp=future_ts, timeframe=Timeframe.DAY)
    sig2 = AlphaEngine.run_model(data_platform, model, [iid], cfg2)[iid]
    assert sig2 < sig1
