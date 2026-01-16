import unittest
from datetime import datetime, timedelta
from typing import Dict
from unittest.mock import MagicMock

import pandas as pd

import src.alpha_library.features  # noqa: F401
from src.core.alpha_engine import FEATURES, AlphaModel
from src.core.data_platform import Bar, DataPlatform


class TestAlphaEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform()
        self.iid = self.data.get_internal_id("AAPL")
        # Need at least 10 bars for SMA_10
        for i in range(15):
            ts = datetime(2025, 1, 1) + timedelta(days=i)
            self.data.add_bars(
                [Bar(self.iid, ts, 100, 101, 99, 100 + i, 1000)]
            )

    def test_feature_registry(self) -> None:
        self.assertIn("sma_10", FEATURES)
        self.assertIn("rsi_14", FEATURES)
        self.assertIn("returns_raw", FEATURES)
        self.assertIn("returns_residual", FEATURES)

    def test_alpha_model_signals(self) -> None:
        class SimpleModel(AlphaModel):
            def compute_signals(
                self, latest: pd.DataFrame, history: pd.DataFrame
            ) -> Dict[int, float]:
                return {iid: 0.1 for iid in latest.index}

        model = SimpleModel(self.data, features=["sma_10"])
        ts = datetime(2025, 1, 1) + timedelta(days=14)
        forecasts = model.generate_forecasts([self.iid], ts)

        self.assertEqual(len(forecasts), 1)
        self.assertEqual(forecasts[self.iid], 0.1)

    def test_alpha_model_not_implemented(self) -> None:
        """Verify base class raises NotImplementedError."""
        model = AlphaModel(MagicMock(), [])
        with self.assertRaises(NotImplementedError):
            model.compute_signals(MagicMock(), MagicMock())

    def test_signal_processor(self) -> None:
        from src.core.alpha_engine import SignalProcessor

        # Test Z-Score
        signals = {1: 10.0, 2: 20.0, 3: 30.0}
        z = SignalProcessor.zscore(signals)
        self.assertAlmostEqual(z[1], -1.22474487, places=6)
        self.assertAlmostEqual(z[3], 1.22474487, places=6)

        # Test Winsorize
        signals = {1: 10.0, 2: -10.0, 3: 0.0}
        w = SignalProcessor.winsorize(signals, limit=3.0)
        self.assertEqual(w[1], 3.0)
        self.assertEqual(w[2], -3.0)

        # Test Decay
        t0 = datetime(2025, 1, 1, 12, 0)
        t1 = datetime(2025, 1, 1, 12, 30)
        # Exponential: half-life 30 mins -> should be 0.5
        d_exp = SignalProcessor.apply_decay(1.0, t0, t1, 30.0)
        self.assertAlmostEqual(d_exp, 0.5)

        # Linear: duration 60 mins, 30 mins elapsed -> should be 0.5
        d_lin = SignalProcessor.apply_linear_decay(1.0, t0, t1, 60.0)
        self.assertAlmostEqual(d_lin, 0.5)

        # Test Rank Transform
        signals = {1: 10.0, 2: 50.0, 3: 20.0}
        # Ranks: 1:1, 2:3, 3:2 -> Pct Ranks: 1:0.33, 2:1.0, 3:0.66
        r = SignalProcessor.rank_transform(signals)
        self.assertAlmostEqual(r[1], 1 / 3)
        self.assertAlmostEqual(r[2], 1.0)


if __name__ == "__main__":
    unittest.main()
