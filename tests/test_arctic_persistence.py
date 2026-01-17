import unittest
from datetime import datetime

from src.core.data_platform import Bar, DataPlatform
from src.core.types import Timeframe


class TestArcticPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "./.arctic_test_db"
        self.data = DataPlatform(db_path=self.db_path, clear=True)

    def test_bar_persistence(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0)
        iid = 1000
        bar = Bar(iid, ts, 100, 101, 99, 100, 1000, timeframe=Timeframe.DAY)
        self.data.add_bars([bar])

        # Create new platform instance to verify persistence
        new_data = DataPlatform(db_path=self.db_path, clear=False)
        from src.core.types import QueryConfig

        df = new_data.get_bars(
            [iid], QueryConfig(start=ts, end=ts, timeframe=Timeframe.DAY)
        )
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["close_1D"], 100.0)

    def test_metadata_persistence(self) -> None:
        self.data.register_security("AAPL", internal_id=5000)
        new_data = DataPlatform(db_path=self.db_path, clear=False)
        iid = new_data.get_internal_id("AAPL")
        self.assertEqual(iid, 5000)


if __name__ == "__main__":
    unittest.main()
