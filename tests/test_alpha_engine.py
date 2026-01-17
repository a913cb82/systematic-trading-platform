import unittest
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

import src.alpha_library.features  # noqa: F401
from src.core.alpha_engine import (
    FEATURES,
    AlphaEngine,
    AlphaModel,
    ModelRunConfig,
    SignalProcessor,
)
from src.core.data_platform import Bar, DataPlatform


class TestAlphaEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)
        self.iid = self.data.get_internal_id("AAPL")
        bars = []
        for i in range(15):
            ts = datetime(2025, 1, 1) + timedelta(days=i)
            # Default timeframe is 1D
            bars.append(Bar(self.iid, ts, 100, 101, 99, 100 + i, 1000))
        self.data.add_bars(bars)

    def test_feature_registry(self) -> None:
        self.assertIn("returns_raw_30min", FEATURES)
        self.assertIn("returns_residual_30min", FEATURES)
        self.assertIn("residual_mom_10_30min", FEATURES)

    def test_alpha_model_signals(self) -> None:
        class SimpleModel(AlphaModel):
            def __init__(self) -> None:
                super().__init__()
                self.feature_names = ["returns_raw_30min"]

            def compute_signals(
                self, latest: pd.DataFrame
            ) -> Dict[int, float]:
                return {iid: 0.1 for iid in latest.index}

        model = SimpleModel()
        ts = datetime(2025, 1, 1, 10, 0)
        # Add data with 30min timeframe
        self.data.add_bars(
            [
                Bar(self.iid, ts, 100, 101, 99, 100, 1000, timeframe="30min"),
                Bar(
                    self.iid,
                    ts - timedelta(minutes=30),
                    99,
                    100,
                    98,
                    99,
                    1000,
                    timeframe="30min",
                ),
            ]
        )

        forecasts = AlphaEngine.run_model(
            self.data,
            model,
            [self.iid],
            ModelRunConfig(timestamp=ts, timeframe="30min"),
        )

        self.assertEqual(len(forecasts), 1)
        self.assertAlmostEqual(forecasts[self.iid], 0.1)

    def test_alpha_model_not_implemented(self) -> None:
        """Verify base class cannot be instantiated."""
        with self.assertRaises(TypeError):
            AlphaModel()  # type: ignore[abstract]

    def test_signal_processor(self) -> None:
        # Test Z-Score
        signals = {1: 10.0, 2: 20.0, 3: 30.0}
        z = SignalProcessor.zscore(signals)
        self.assertAlmostEqual(z[1], -1.22474487, places=6)
        self.assertAlmostEqual(z[3], 1.22474487, places=6)


if __name__ == "__main__":
    unittest.main()
