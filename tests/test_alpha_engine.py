import unittest
from datetime import datetime, timedelta
from typing import Dict
from unittest.mock import MagicMock

import pandas as pd

from src.alpha_engine import FEATURES, AlphaModel
from src.data_platform import Bar, DataPlatform


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

    def test_alpha_model_signals(self) -> None:
        class SimpleModel(AlphaModel):
            def compute_signals(
                self, latest: pd.DataFrame, returns: pd.DataFrame
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


if __name__ == "__main__":
    unittest.main()
