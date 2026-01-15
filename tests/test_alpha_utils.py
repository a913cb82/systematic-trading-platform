import unittest
import os
from datetime import datetime
from src.alpha.processing import SignalProcessor
from src.alpha.combiner import SignalCombiner
from src.alpha.publisher import ForecastPublisher


class TestAlphaUtils(unittest.TestCase):
    def test_processing(self):
        forecasts = {1: 1.0, 2: 2.0, 3: 3.0}
        z = SignalProcessor.z_score(forecasts)
        self.assertAlmostEqual(z[1], -1.224744871391589)
        self.assertAlmostEqual(z[2], 0.0)
        self.assertAlmostEqual(z[3], 1.224744871391589)

        w = SignalProcessor.winsorize({1: 10.0, 2: -10.0, 3: 0.0}, limit=3.0)
        self.assertEqual(w[1], 3.0)
        self.assertEqual(w[2], -3.0)
        self.assertEqual(w[3], 0.0)

    def test_combiner(self):
        f1 = {1: 1.0, 2: 0.0}
        f2 = {1: 0.5, 2: 1.0}
        combined = SignalCombiner.equal_weight([f1, f2])
        self.assertEqual(combined[1], 0.75)
        self.assertEqual(combined[2], 0.5)

    def test_publisher(self):
        db_path = "test_forecasts.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        pub = ForecastPublisher(db_path)

        ts = datetime(2023, 1, 1, 12)
        f = {1: 0.5, 2: -0.2}
        pub.submit_forecasts(ts, f)

        read_f = pub.get_forecasts(ts)
        self.assertEqual(read_f[1], 0.5)
        self.assertEqual(read_f[2], -0.2)

        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    unittest.main()
