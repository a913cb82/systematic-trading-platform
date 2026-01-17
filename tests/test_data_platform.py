import unittest
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from src.core.data_platform import Bar, CorporateAction, DataPlatform
from src.gateways.base import DataProvider


class TestDataPlatformFull(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)
        self.iid = self.data.get_internal_id("AAPL")
        self.ts = datetime(2025, 1, 1, 9, 30)

    def test_ism_and_symbology(self) -> None:
        self.assertEqual(self.data.get_internal_id("AAPL"), 1000)
        self.assertEqual(self.data.get_internal_id("MSFT"), 1001)
        self.assertEqual(self.data.reverse_ism[1000], "AAPL")

    def test_bitemporal_logic(self) -> None:
        # Bar at T, known at T+1
        self.data.add_bars(
            [
                Bar(
                    1000,
                    self.ts,
                    100,
                    101,
                    99,
                    100,
                    1000,
                    timestamp_knowledge=self.ts + timedelta(minutes=1),
                )
            ]
        )
        # Correction at T, known at T+5
        self.data.add_bars(
            [
                Bar(
                    1000,
                    self.ts,
                    100,
                    101,
                    99,
                    102,
                    1000,
                    timestamp_knowledge=self.ts + timedelta(minutes=5),
                )
            ]
        )

        from src.core.data_platform import QueryConfig

        # Query at T+2: should see 100
        df = self.data.get_bars(
            [1000],
            QueryConfig(
                start=self.ts,
                end=self.ts,
                timeframe="1D",
                as_of=self.ts + timedelta(minutes=2),
            ),
        )
        self.assertEqual(df.iloc[0]["close_1D"], 100)

        # Query at T+6: should see 102
        df = self.data.get_bars(
            [1000],
            QueryConfig(
                start=self.ts,
                end=self.ts,
                timeframe="1D",
                as_of=self.ts + timedelta(minutes=6),
            ),
        )
        self.assertEqual(df.iloc[0]["close_1D"], 102)

    def test_corporate_actions_adjustment(self) -> None:
        from src.core.data_platform import QueryConfig

        ts1, ts2 = self.ts, self.ts + timedelta(days=1)
        self.data.add_bars(
            [
                Bar(1000, ts1, 100, 100, 100, 100, 1000),
                Bar(1000, ts2, 50, 50, 50, 50, 1000),
            ]
        )
        self.data.add_ca(CorporateAction(1000, ts2, "SPLIT", 2.0))

        # Ratio adjusted: ts1 100 -> 50
        df = self.data.get_bars(
            [1000],
            QueryConfig(start=ts1, end=ts2, timeframe="1D", adjust=True),
        )
        self.assertEqual(df[df["timestamp"] == ts1].iloc[0]["close_1D"], 50)

    def test_sync_data_with_corporate_actions(self) -> None:
        """Test that sync_data correctly ingests CA from provider."""

        class MockProvider(DataProvider):
            def fetch_bars(
                self, tickers: List[str], start: datetime, end: datetime
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "ticker": ["AAPL"],
                        "timestamp": [start],
                        "open": [100],
                        "high": [101],
                        "low": [99],
                        "close": [100],
                        "volume": [1000],
                    }
                )

            def fetch_corporate_actions(
                self, tickers: List[str], start: datetime, end: datetime
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "ticker": ["AAPL"],
                        "ex_date": [start],
                        "type": ["SPLIT"],
                        "ratio": [2.0],
                    }
                )

        dp = DataPlatform(MockProvider(), clear=True)
        dp.sync_data(["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 2))
        self.assertFalse(dp.ca_df.empty)
        self.assertEqual(dp.ca_df.iloc[0]["ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
