import os
import shutil
import unittest
from datetime import datetime
from typing import List

from src.common.types import Bar
from src.data.market_data import MarketDataEngine


class TestMarketDataEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = "test_market_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.engine = MarketDataEngine(self.test_dir)

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_write_and_read_bars(self) -> None:
        bars = [
            Bar(
                internal_id=1,
                timestamp=datetime(2023, 1, 1, 9, 30),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            ),
            Bar(
                internal_id=1,
                timestamp=datetime(2023, 1, 1, 9, 31),
                open=100.5,
                high=102.0,
                low=100.5,
                close=101.5,
                volume=1500.0,
            ),
            Bar(
                internal_id=2,
                timestamp=datetime(2023, 1, 1, 9, 30),
                open=50.0,
                high=51.0,
                low=49.0,
                close=50.5,
                volume=500.0,
            ),
        ]

        self.engine.write_bars(bars)

        # Read back bars for ID 1
        read_bars = self.engine.get_bars(
            [1], datetime(2023, 1, 1, 9, 30), datetime(2023, 1, 1, 9, 31)
        )
        self.assertEqual(len(read_bars), 2)
        self.assertEqual(read_bars[0]["internal_id"], 1)
        self.assertEqual(read_bars[1]["close"], 101.5)

        # Read back bars for both IDs
        all_read_bars = self.engine.get_bars(
            [1, 2], datetime(2023, 1, 1, 9, 30), datetime(2023, 1, 1, 9, 31)
        )
        self.assertEqual(len(all_read_bars), 3)

    def test_subscribe_bars(self) -> None:
        received_bars: List[Bar] = []

        def on_bar(bar: Bar) -> None:
            received_bars.append(bar)

        self.engine.subscribe_bars([1], on_bar)

        bars = [
            Bar(
                internal_id=1,
                timestamp=datetime(2023, 1, 1, 9, 30),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            ),
            Bar(
                internal_id=2,
                timestamp=datetime(2023, 1, 1, 9, 30),
                open=50.0,
                high=51.0,
                low=49.0,
                close=50.5,
                volume=500.0,
            ),
        ]

        self.engine.write_bars(bars)

        self.assertEqual(len(received_bars), 1)
        self.assertEqual(received_bars[0]["internal_id"], 1)

    def test_get_universe(self) -> None:
        bars = [
            Bar(
                internal_id=1,
                timestamp=datetime(2023, 1, 1, 9, 30),
                timestamp_knowledge=datetime(2023, 1, 1, 9, 30),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            ),
            Bar(
                internal_id=3,
                timestamp=datetime(2023, 1, 1, 9, 30),
                timestamp_knowledge=datetime(2023, 1, 1, 9, 30),
                open=50.0,
                high=51.0,
                low=49.0,
                close=50.5,
                volume=500.0,
            ),
        ]
        self.engine.write_bars(bars)

        universe = self.engine.get_universe(datetime(2023, 1, 1))
        self.assertIn(1, universe)
        self.assertIn(3, universe)
        self.assertEqual(len(universe), 2)

    def test_bitemporal_bars(self) -> None:
        # Initial version of a bar
        bar_v1 = Bar(
            internal_id=1,
            timestamp=datetime(2023, 1, 1, 9, 30),
            timestamp_knowledge=datetime(2023, 1, 1, 9, 31),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
        )
        self.engine.write_bars([bar_v1])

        # Corrected version of the same bar, known later
        bar_v2 = Bar(
            internal_id=1,
            timestamp=datetime(2023, 1, 1, 9, 30),
            timestamp_knowledge=datetime(2023, 1, 1, 9, 35),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.6,
            volume=1000.0,
        )
        self.engine.write_bars([bar_v2])

        # Query as of 9:32 (should get v1)
        bars_932 = self.engine.get_bars(
            [1],
            datetime(2023, 1, 1, 9, 30),
            datetime(2023, 1, 1, 9, 30),
            as_of=datetime(2023, 1, 1, 9, 32),
        )
        self.assertEqual(len(bars_932), 1)
        self.assertEqual(bars_932[0]["close"], 100.5)

        # Query as of 9:36 (should get v2)
        bars_936 = self.engine.get_bars(
            [1],
            datetime(2023, 1, 1, 9, 30),
            datetime(2023, 1, 1, 9, 30),
            as_of=datetime(2023, 1, 1, 9, 36),
        )
        self.assertEqual(len(bars_936), 1)
        self.assertEqual(bars_936[0]["close"], 100.6)


if __name__ == "__main__":
    unittest.main()
