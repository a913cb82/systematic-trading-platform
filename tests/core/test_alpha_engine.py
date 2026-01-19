import threading
import time
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import pytest

from src.core.alpha_engine import (
    AlphaEngine,
    AlphaModel,
    ModelRunConfig,
    SignalCombiner,
    SignalProcessor,
    alpha_context,
)
from src.core.data_platform import Bar, DataPlatform
from src.core.types import QueryConfig, Timeframe

# Constants to avoid magic values
AAPL_IID = 1000
TEST_SIGNALS = {1: 10.0, 2: 20.0, 3: 30.0}
ZERO_STD_SIGNALS = {1: 10.0, 2: 10.0}
SIGNAL_WEIGHTS = [0.6, 0.4]
LOOKBACK_BARS = 25
W1 = 0.6
W2 = 0.9


class MockModel(AlphaModel):
    def __init__(self) -> None:
        super().__init__()
        self.feature_names = ["sma_20_30min"]

    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        res = {
            int(idx): float(row["sma_20_30min"])
            for idx, row in latest.iterrows()
        }
        return res


def test_alpha_engine_calculates_signals_from_hydrated_features(
    populated_platform: Any,
) -> None:
    data, iid, ts = populated_platform
    model = MockModel()
    config = ModelRunConfig(timestamp=ts, timeframe=Timeframe.MIN_30)
    signals = AlphaEngine.run_model(data, model, [iid], config)

    assert iid in signals
    assert isinstance(signals[iid], float)


def test_signal_processor_standardizes_to_zscores() -> None:
    assert SignalProcessor.zscore({}) == {}
    z = SignalProcessor.zscore(TEST_SIGNALS)
    assert pytest.approx(z[2]) == 0.0
    assert z[3] > 0
    assert z[1] < 0


def test_signal_processor_handles_zero_volatility_signals() -> None:
    assert SignalProcessor.zscore(ZERO_STD_SIGNALS) == {1: 0.0, 2: 0.0}


def test_signal_combiner_aggregates_multiple_models_with_weights() -> None:
    assert SignalCombiner.combine([]) == {}
    s1, s2 = {1: 1.0, 2: 0.5}, {1: 0.0, 2: 1.5}
    combined = SignalCombiner.combine([s1, s2], weights=SIGNAL_WEIGHTS)
    assert pytest.approx(combined[1]) == W1
    assert pytest.approx(combined[2]) == W2


def test_alpha_engine_isolates_execution_context_per_thread(
    data_platform: DataPlatform,
) -> None:
    class ContextModel(AlphaModel):
        def compute_signals(self, latest: pd.DataFrame) -> Dict[int, Any]:
            return {0: self.context_as_of}

    ts1, ts2 = datetime(2025, 1, 1), datetime(2025, 2, 2)
    model = ContextModel()
    results: Dict[str, Any] = {}

    def run_model(ts: datetime, key: str) -> None:
        with alpha_context(data_platform, ts):
            time.sleep(0.1)
            results[key] = model.compute_signals(pd.DataFrame())[0]

    t1 = threading.Thread(target=run_model, args=(ts1, "one"))
    t2 = threading.Thread(target=run_model, args=(ts2, "two"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["one"] == ts1
    assert results["two"] == ts2


def test_alpha_engine_recursively_hydrates_feature_dependencies(
    data_platform: DataPlatform,
) -> None:
    ts = datetime(2025, 1, 1, 12, 0)
    iid = data_platform.register_security("AAPL")
    bars = []
    for i in range(LOOKBACK_BARS):
        delta = pd.Timedelta(minutes=30 * i)
        bar = Bar(
            iid,
            ts - delta,
            100,
            101,
            99,
            100,
            1000,
            timeframe=Timeframe.MIN_30,
        )
        bars.append(bar)
    data_platform.add_bars(bars)

    query = QueryConfig(
        start=ts - pd.Timedelta(days=1), end=ts, timeframe=Timeframe.MIN_30
    )
    df = data_platform.get_bars([iid], query)
    AlphaEngine._hydrate_features(df, ["residual_vol_20_30min"])
    assert "returns_raw_30min" in df.columns
    assert "returns_residual_30min" in df.columns
    assert "residual_vol_20_30min" in df.columns


def test_alpha_engine_gracefully_handles_empty_bar_histories(
    data_platform: DataPlatform,
) -> None:
    class EmptyModel(AlphaModel):
        def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
            return {}

    res = AlphaEngine.run_model(
        data_platform, EmptyModel(), [999], ModelRunConfig(datetime.now())
    )
    assert res == {}
