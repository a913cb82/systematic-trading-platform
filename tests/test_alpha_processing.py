import unittest

from src.alpha.processing import SignalProcessor


class TestSignalProcessor(unittest.TestCase):
    def test_winsorize(self):
        forecasts = {1: 10.0, 2: -5.0, 3: 0.5}
        winsorized = SignalProcessor.winsorize(forecasts, limit=3.0)
        self.assertEqual(winsorized[1], 3.0)
        self.assertEqual(winsorized[2], -3.0)
        self.assertEqual(winsorized[3], 0.5)

    def test_apply_decay(self):
        initial_val = 1.0
        half_life = 10.0

        # After 10 units, it should be 0.5
        val_10 = SignalProcessor.apply_decay(initial_val, 0, 10, half_life)
        self.assertAlmostEqual(val_10, 0.5)

        # After 20 units, it should be 0.25
        val_20 = SignalProcessor.apply_decay(initial_val, 0, 20, half_life)
        self.assertAlmostEqual(val_20, 0.25)


if __name__ == "__main__":
    unittest.main()
