import unittest
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd

from src.alpha_engine import FEATURES, AlphaModel, SignalProcessor
from src.data_platform import Bar, DataPlatform


class TestAlphaSophistication(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform()
        self.iid = self.data.get_internal_id("AAPL")
        # Generate 40 days of price data for technical indicators
        for i in range(40):
            ts = datetime(2025, 1, 1) + timedelta(days=i)
            # Create a simple trend with some noise
            price = 100 + i * 0.5 + np.sin(i)
            self.data.add_bars(
                [Bar(self.iid, ts, price, price + 1, price - 1, price, 1000)]
            )

    def test_signal_processing(self) -> None:
        """Tests Z-scoring and Winsorization."""
        signals = {1000: 10.0, 1001: 2.0, 1002: 0.0, 1003: -5.0}

        # 1. Z-Score
        z = SignalProcessor.zscore(signals)
        vals = list(z.values())
        self.assertAlmostEqual(float(np.mean(vals)), 0.0)
        self.assertAlmostEqual(float(np.std(vals)), 1.0)

        # 2. Winsorize (at 1.0 std)
        w = SignalProcessor.winsorize(z, limit=1.0)
        self.assertAlmostEqual(max(w.values()), 1.0)
        self.assertAlmostEqual(min(w.values()), -1.0)

    def test_macd_feature(self) -> None:
        """Tests MACD calculation hydration."""

        class MACDModel(AlphaModel):
            def compute_signals(
                self, latest: pd.DataFrame, returns: pd.DataFrame
            ) -> Dict[int, float]:
                # Convert the 'latest' DataFrame row to a dict of floats
                return {
                    int(idx): float(row["macd"])
                    for idx, row in latest.iterrows()
                }

        model = MACDModel(self.data, features=["macd"])
        ts = datetime(2025, 1, 1) + timedelta(days=39)
        forecasts = model.generate_forecasts([self.iid], ts)

        self.assertIn(self.iid, forecasts)
        self.assertIsInstance(forecasts[self.iid], float)

    def test_ofi_feature(self) -> None:
        """Tests Order Flow Imbalance logic."""
        ts = datetime(2025, 2, 20)
        # Up bar -> Positive OFI
        _ = Bar(self.iid, ts, 100, 105, 95, 102, 1000)
        # Down bar -> Negative OFI
        _ = Bar(self.iid, ts + timedelta(minutes=1), 102, 105, 95, 100, 500)

        df = pd.DataFrame(
            [
                {
                    "internal_id": self.iid,
                    "open": 100,
                    "close": 102,
                    "volume": 1000,
                },
                {
                    "internal_id": self.iid,
                    "open": 102,
                    "close": 100,
                    "volume": 500,
                },
            ]
        )

        ofi_vals = FEATURES["ofi"](df)
        self.assertEqual(ofi_vals.iloc[0], 1000)
        self.assertEqual(ofi_vals.iloc[1], -500)

    def test_empty_signals_handling(self) -> None:
        """Tests that processing doesn't crash on empty input."""
        self.assertEqual(SignalProcessor.zscore({}), {})
        self.assertEqual(SignalProcessor.winsorize({}), {})
        self.assertEqual(SignalProcessor.rank_transform({}), {})

    def test_signal_decay(self) -> None:
        """Tests exponential decay calculation."""
        t0 = datetime(2025, 1, 1, 12, 0)
        t1 = t0 + timedelta(minutes=60)
        # 1.0 signal with 60 min half-life should be 0.5 after 60 mins
        res = SignalProcessor.apply_decay(1.0, t0, t1, 60.0)
        self.assertAlmostEqual(res, 0.5)

    def test_rank_transform(self) -> None:
        """Tests percentile ranking of signals."""
        signals = {1000: 10.0, 1001: 20.0, 1002: 30.0, 1003: 40.0}
        ranks = SignalProcessor.rank_transform(signals)
        # Should be 0.25, 0.5, 0.75, 1.0
        self.assertEqual(ranks[1000], 0.25)
        self.assertEqual(ranks[1003], 1.0)


if __name__ == "__main__":
    unittest.main()
