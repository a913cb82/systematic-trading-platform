import unittest
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from src.core.data_platform import DataPlatform
from src.core.types import Bar, QueryConfig, Timeframe
from src.gateways.base import BarProvider, CorporateActionProvider


class TestLiveDemoBugs(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(db_path="./.arctic_test_bugs", clear=True)
        self.ts = datetime(2025, 1, 1, 12, 0)
        # Register AAPL
        self.iid = self.data.register_security("AAPL")

    def test_add_bars_resolves_ticker(self) -> None:
        """
        Streaming bars come with internal_id=0 but have _ticker.
        DataPlatform should resolve this using the ISM.
        """
        b = Bar(
            internal_id=0,
            timestamp=self.ts,
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1000,
            timeframe=Timeframe.MINUTE,
        )
        b._ticker = "AAPL"

        self.data.add_bars([b])

        df = self.data.get_bars(
            [self.iid],
            QueryConfig(
                start=self.ts, end=self.ts, timeframe=Timeframe.MINUTE
            ),
        )
        self.assertFalse(
            df.empty, "Data should have been written under correct internal_id"
        )
        self.assertEqual(df.iloc[0]["close_1min"], 100.0)

    def test_write_deduplication_integrity(self) -> None:
        """
        Bitemporal integrity: keep both versions in DB,
        but query returns latest.
        """
        t1 = self.ts
        tk1 = self.ts + timedelta(minutes=1)
        tk2 = self.ts + timedelta(minutes=2)

        # Original version
        b1 = Bar(
            self.iid,
            t1,
            100,
            101,
            99,
            100,
            1000,
            timeframe=Timeframe.MINUTE,
            timestamp_knowledge=tk1,
        )
        self.data.add_bars([b1])

        # Correction version
        b2 = Bar(
            self.iid,
            t1,
            100,
            105,
            99,
            105,
            1000,
            timeframe=Timeframe.MINUTE,
            timestamp_knowledge=tk2,
        )
        self.data.add_bars([b2])

        # Query normally should get LATEST version (tk2)
        df = self.data.get_bars(
            [self.iid],
            QueryConfig(start=t1, end=t1, timeframe=Timeframe.MINUTE),
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["close_1min"], 105.0)

        # Verify database size - MUST be 2 if we want bitemporal history
        raw = self.data.lib.read("bars").data
        self.assertEqual(
            len(raw),
            2,
            "Database should preserve both versions for bitemporal integrity",
        )

    def test_sync_data_resolves_ids(self) -> None:
        """Test that sync_data correctly maps tickers to internal IDs."""

        class MockProv(BarProvider, CorporateActionProvider):
            def fetch_bars(
                self,
                tickers: List[str],
                start: datetime,
                end: datetime,
                timeframe: Timeframe = Timeframe.DAY,
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "ticker": "AAPL",
                            "timestamp": start,
                            "open": 100,
                            "high": 101,
                            "low": 99,
                            "close": 100,
                            "volume": 1000,
                        }
                    ]
                )

            def fetch_corporate_actions(
                self, tickers: List[str], start: datetime, end: datetime
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "ticker": "AAPL",
                            "ex_date": start,
                            "type": "DIV",
                            "value": 1.0,
                        }
                    ]
                )

        dp = DataPlatform(
            MockProv(), db_path="./.arctic_test_sync", clear=True
        )
        # Register AAPL manually first so we know its ID
        expected_iid = dp.register_security("AAPL")

        dp.sync_data(["AAPL"], self.ts, self.ts)

        # Check bars
        bars = dp.lib.read("bars").data
        self.assertEqual(bars.iloc[0]["internal_id"], expected_iid)

        # Check corporate actions
        ca = dp.ca_df
        self.assertEqual(ca.iloc[0]["internal_id"], expected_iid)


if __name__ == "__main__":
    unittest.main()
