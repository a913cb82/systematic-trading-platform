import shutil
import tempfile
import unittest
from datetime import datetime

from src.common.types import Bar
from src.data.market_data import MarketDataEngine


class TestBarValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.mde = MarketDataEngine(self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_invalid_ohlc(self) -> None:
        # High < Low
        invalid_bar = Bar(
            internal_id=1,
            timestamp=datetime(2023, 1, 1),
            open=100.0,
            high=90.0,
            low=110.0,
            close=100.0,
            volume=1000.0,
        )
        self.mde.write_bars([invalid_bar])
        bars = self.mde.get_bars(
            [1], datetime(2023, 1, 1), datetime(2023, 1, 1)
        )
        self.assertEqual(len(bars), 0)

    def test_negative_volume(self) -> None:
        invalid_bar = Bar(
            internal_id=1,
            timestamp=datetime(2023, 1, 1),
            open=100.0,
            high=110.0,
            low=90.0,
            close=100.0,
            volume=-100.0,
        )
        self.mde.write_bars([invalid_bar])
        bars = self.mde.get_bars(
            [1], datetime(2023, 1, 1), datetime(2023, 1, 1)
        )
        self.assertEqual(len(bars), 0)

    def test_gap_filling(self) -> None:
        # Day 1, 9:30
        bar1 = Bar(
            internal_id=1,
            timestamp=datetime(2023, 1, 1, 9, 30),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
        )
        # Day 1, 9:32 (Missing 9:31)
        bar2 = Bar(
            internal_id=1,
            timestamp=datetime(2023, 1, 1, 9, 32),
            open=100.5,
            high=102.0,
            low=100.0,
            close=101.0,
            volume=1000.0,
        )

        self.mde.write_bars([bar1])
        # Write second bar with fill_gaps=True
        self.mde.write_bars([bar2], fill_gaps=True)

        # Should now have 3 bars: 9:30, 9:31 (synthetic), 9:32
        bars = self.mde.get_bars(
            [1], datetime(2023, 1, 1, 9, 30), datetime(2023, 1, 1, 9, 32)
        )
        self.assertEqual(len(bars), 3)
        self.assertEqual(bars[1]["timestamp"], datetime(2023, 1, 1, 9, 31))
        self.assertEqual(bars[1]["volume"], 0.0)
        self.assertEqual(bars[1]["close"], 100.5)  # Carried from bar1


if __name__ == "__main__":
    unittest.main()
