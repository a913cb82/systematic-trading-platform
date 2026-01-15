import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

from src.common.types import Bar
from src.data.market_data import MarketDataEngine


class TestReturnsEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mde = MarketDataEngine(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_raw_returns(self):
        # Day 1: 100
        # Day 2: 110 (10% return)
        # Day 3: 121 (10% return)
        internal_id = 1
        start_date = datetime(2023, 1, 1)
        bars = []
        for i in range(3):
            ts = start_date + timedelta(days=i)
            price = 100.0 * (1.1**i)
            bars.append(
                Bar(
                    internal_id=internal_id,
                    timestamp=ts,
                    close=price,
                    volume=1000.0,
                )
            )
        self.mde.write_bars(bars)

        returns = self.mde.get_returns(
            [internal_id], start_date, start_date + timedelta(days=2)
        )

        self.assertEqual(len(returns), 2)
        self.assertAlmostEqual(returns.iloc[0, 0], 0.1)
        self.assertAlmostEqual(returns.iloc[1, 0], 0.1)


if __name__ == "__main__":
    unittest.main()
