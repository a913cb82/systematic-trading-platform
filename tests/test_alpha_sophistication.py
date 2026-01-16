import unittest

import numpy as np

import src.alpha_library.features  # noqa: F401
from src.core.alpha_engine import SignalProcessor
from src.core.data_platform import DataPlatform


class TestAlphaSophistication(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform()
        self.iid = self.data.get_internal_id("AAPL")

    def test_signal_processing(self) -> None:
        """Tests Z-scoring."""
        signals = {1000: 10.0, 1001: 2.0, 1002: 0.0, 1003: -5.0}

        # 1. Z-Score
        z = SignalProcessor.zscore(signals)
        vals = list(z.values())
        self.assertAlmostEqual(float(np.mean(vals)), 0.0)
        self.assertAlmostEqual(float(np.std(vals)), 1.0)

    def test_empty_signals_handling(self) -> None:
        """Tests that processing doesn't crash on empty input."""
        self.assertEqual(SignalProcessor.zscore({}), {})


if __name__ == "__main__":
    unittest.main()
