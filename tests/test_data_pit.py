import unittest
from datetime import datetime, timedelta

from src.core.data_platform import Bar, DataPlatform
from src.core.types import QueryConfig, Timeframe


class TestDataPIT(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)
        self.iid = self.data.register_security("AAPL")

    def test_pit_bars(self) -> None:
        """Tests that bars are retrieved correctly based on knowledge time."""
        t1 = datetime(2025, 1, 1, 10, 0)
        t2 = datetime(2025, 1, 1, 11, 0)

        # First version of bar
        b1 = Bar(self.iid, t1, 100, 101, 99, 100, 1000, timestamp_knowledge=t1)
        self.data.add_bars([b1])

        # Restated version of same bar
        b1_v2 = Bar(
            self.iid, t1, 100, 105, 99, 105, 1000, timestamp_knowledge=t2
        )
        self.data.add_bars([b1_v2])

        # Query as of t1 (should get v1)
        df1 = self.data.get_bars(
            [self.iid],
            QueryConfig(start=t1, end=t1, timeframe=Timeframe.DAY, as_of=t1),
        )
        self.assertEqual(df1.iloc[0]["close_1D"], 100.0)

        # Query as of t2 (should get v2)
        df2 = self.data.get_bars(
            [self.iid],
            QueryConfig(start=t1, end=t1, timeframe=Timeframe.DAY, as_of=t2),
        )
        self.assertEqual(df2.iloc[0]["close_1D"], 105.0)

    def test_pit_universe(self) -> None:
        """Tests universe reconstruction at specific dates."""
        t1 = datetime(2025, 1, 1)
        t2 = datetime(2025, 2, 1)

        # AAPL exists from t1
        self.data.register_security("AAPL", start=t1)
        # MSFT exists from t2
        self.data.register_security("MSFT", start=t2)

        u1 = self.data.get_universe(t1 + timedelta(days=1))
        self.assertEqual(len(u1), 1)

        u2 = self.data.get_universe(t2 + timedelta(days=1))
        self.assertEqual(len(u2), 2)


if __name__ == "__main__":
    unittest.main()
