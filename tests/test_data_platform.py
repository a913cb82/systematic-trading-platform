import unittest
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from src.base import DataProvider
from src.data_platform import Bar, CorporateAction, DataPlatform


class TestDataPlatformFull(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform()
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

        # Query at T+2: should see 100
        df = self.data.get_bars(
            [1000], self.ts, self.ts, as_of=self.ts + timedelta(minutes=2)
        )
        self.assertEqual(df.iloc[0]["close"], 100)

        # Query at T+6: should see 102
        df = self.data.get_bars(
            [1000], self.ts, self.ts, as_of=self.ts + timedelta(minutes=6)
        )
        self.assertEqual(df.iloc[0]["close"], 102)

    def test_corporate_actions_adjustment(self) -> None:
        ts1, ts2 = self.ts, self.ts + timedelta(days=1)
        self.data.add_bars(
            [
                Bar(1000, ts1, 100, 100, 100, 100, 1000),
                Bar(1000, ts2, 50, 50, 50, 50, 1000),
            ]
        )
        self.data.add_ca(CorporateAction(1000, ts2, "SPLIT", 2.0))

        # Ratio adjusted: ts1 100 -> 50
        df = self.data.get_bars([1000], ts1, ts2, adjust=True)
        self.assertEqual(df[df["timestamp"] == ts1].iloc[0]["close"], 50)

    def test_validation_and_gap_filling(self) -> None:
        # 1. Validation
        invalid = [
            Bar(1000, self.ts, 100, 90, 110, 100, 1000),  # High < Low
            Bar(1000, self.ts, 100, 110, 90, 100, -1),
        ]  # Neg Vol
        self.data.add_bars(invalid)
        self.assertEqual(len(self.data.get_bars([1000], self.ts, self.ts)), 0)

        # 2. Gap Filling
        t1 = self.ts
        t3 = self.ts + timedelta(minutes=2)  # Gap at t2
        self.data.add_bars(
            [
                Bar(1000, t1, 100, 100, 100, 100, 1000),
                Bar(1000, t3, 101, 101, 101, 101, 1000),
            ],
            fill_gaps=True,
        )
        df = self.data.get_bars([1000], t1, t3)
        self.assertEqual(len(df), 3)
        self.assertEqual(
            df.iloc[1]["timestamp"], self.ts + timedelta(minutes=1)
        )
        self.assertEqual(df.iloc[1]["volume"], 0)

    def test_residual_returns(self) -> None:
        t1, t2 = self.ts, self.ts + timedelta(days=1)
        self.data.add_bars(
            [
                Bar(1000, t1, 100, 100, 100, 100, 1000),
                Bar(1000, t2, 101, 101, 101, 101, 1000),
                Bar(999, t1, 100, 100, 100, 100, 1000),
                Bar(999, t2, 102, 102, 102, 102, 1000),
            ]
        )  # Benchmark up 2%

        # Asset up 1%, Bench up 2% -> Residual -1%
        rets = self.data.get_returns([1000], t1, t2, benchmark_id=999)
        self.assertAlmostEqual(rets.iloc[0][1000], -0.01)

    def test_get_returns_empty(self) -> None:
        """Test returns calculation with no data."""
        rets = self.data.get_returns(
            [1000], datetime(2025, 1, 1), datetime(2025, 1, 2)
        )
        self.assertTrue(rets.empty)

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

        dp = DataPlatform(MockProvider())
        dp.sync_data(["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 2))
        self.assertFalse(dp.ca_df.empty)
        self.assertEqual(dp.ca_df.iloc[0]["ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
