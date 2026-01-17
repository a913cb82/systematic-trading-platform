import unittest
from datetime import datetime
from typing import Dict

import pandas as pd

import src.alpha_library.features  # noqa: F401
from src.core.alpha_engine import (
    AlphaEngine,
    AlphaModel,
    ModelRunConfig,
    SignalCombiner,
    SignalProcessor,
)
from src.core.data_platform import Bar, DataPlatform
from src.core.types import Timeframe


class MockModel(AlphaModel):
    def __init__(self) -> None:
        super().__init__()
        self.feature_names = ["sma_20_30min"]

    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        return {
            int(idx): float(row["sma_20_30min"])
            for idx, row in latest.iterrows()
        }


class TestAlphaEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)
        self.ts = datetime(2025, 1, 1, 12, 0)
        self.iid = self.data.register_security("AAPL")

        # Create 30 bars of history
        bars = [
            Bar(
                self.iid,
                self.ts - pd.Timedelta(minutes=30 * i),
                100 + i,
                101 + i,
                99 + i,
                100 + i,
                1000,
                timeframe=Timeframe.MIN_30,
            )
            for i in range(30)
        ]
        self.data.add_bars(bars)

    def test_run_model(self) -> None:
        model = MockModel()
        config = ModelRunConfig(timestamp=self.ts, timeframe=Timeframe.MIN_30)
        signals = AlphaEngine.run_model(self.data, model, [self.iid], config)

        self.assertIn(self.iid, signals)
        self.assertIsInstance(signals[self.iid], float)

    def test_signal_processor_zscore(self) -> None:
        signals = {1: 10.0, 2: 20.0, 3: 30.0}
        z = SignalProcessor.zscore(signals)
        self.assertAlmostEqual(z[2], 0.0)
        self.assertTrue(z[3] > 0)
        self.assertTrue(z[1] < 0)

    def test_signal_combiner(self) -> None:
        s1 = {1: 1.0, 2: 0.5}
        s2 = {1: 0.0, 2: 1.5}
        combined = SignalCombiner.combine([s1, s2], weights=[0.6, 0.4])
        self.assertAlmostEqual(combined[1], 0.6)
        self.assertAlmostEqual(combined[2], 0.9)


if __name__ == "__main__":
    unittest.main()
