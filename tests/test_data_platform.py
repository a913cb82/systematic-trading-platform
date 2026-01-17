import unittest
from datetime import datetime
from typing import List

import pandas as pd

from src.core.data_platform import (
    Bar,
    CorporateAction,
    DataPlatform,
    Event,
)
from src.core.types import QueryConfig, Timeframe
from src.gateways.base import DataProvider


class TestDataPlatform(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)

    def test_register_security(self) -> None:
        iid = self.data.register_security("AAPL", sector="Tech")
        self.assertEqual(self.data.get_internal_id("AAPL"), iid)

        # Check metadata persistence
        secs = self.data.get_securities(["AAPL"])
        self.assertEqual(len(secs), 1)
        self.assertEqual(secs[0].extra["sector"], "Tech")

    def test_add_get_bars(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0)
        iid = self.data.register_security("AAPL")
        bar = Bar(
            iid,
            ts,
            150.0,
            155.0,
            149.0,
            152.0,
            1000000,
            timeframe=Timeframe.DAY,
        )
        self.data.add_bars([bar])

        df = self.data.get_bars(
            [iid], QueryConfig(start=ts, end=ts, timeframe=Timeframe.DAY)
        )
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["close_1D"], 152.0)

    def test_corporate_action_adjustment(self) -> None:
        ts1 = datetime(2025, 1, 1)
        ts2 = datetime(2025, 1, 2)
        iid = self.data.register_security("AAPL")

        # Price 100 before split
        self.data.add_bars(
            [
                Bar(
                    iid,
                    ts1,
                    100,
                    100,
                    100,
                    100,
                    1000,
                    timeframe=Timeframe.DAY,
                )
            ]
        )
        # Split 2:1 on ts2
        self.data.add_ca(CorporateAction(iid, ts2, "SPLIT", 2.0))

        # Adjusted price for ts1 should be 50.
        # We query with end=ts2 so that the split at ts2 is included.
        df = self.data.get_bars(
            [iid],
            QueryConfig(
                start=ts1, end=ts2, timeframe=Timeframe.DAY, adjust=True
            ),
        )
        self.assertEqual(df[df.timestamp == ts1].iloc[0]["close_1D"], 50.0)

    def test_event_storage(self) -> None:
        ts = datetime(2025, 1, 1)
        iid = self.data.register_security("AAPL")
        event = Event(iid, ts, "EARNINGS", {"eps": 1.5})
        self.data.add_events([event])

        events = self.data.get_events([iid], types=["EARNINGS"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].value["eps"], 1.5)


class TestDataPlatformFull(unittest.TestCase):
    def test_sync_data_with_corporate_actions(self) -> None:
        """Test that sync_data correctly ingests CA from provider."""

        class MockProvider(DataProvider):
            def fetch_bars(
                self,
                tickers: List[str],
                start: datetime,
                end: datetime,
                timeframe: Timeframe = Timeframe.DAY,
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
                        "timeframe": [timeframe.value],
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
                        "value": [2.0],
                    }
                )

            def fetch_events(
                self, tickers: List[str], start: datetime, end: datetime
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    columns=["ticker", "timestamp", "event_type", "value"]
                )

        dp = DataPlatform(MockProvider(), clear=True)
        dp.sync_data(
            ["AAPL"],
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
            timeframe=Timeframe.DAY,
        )
        self.assertFalse(dp.ca_df.empty)
        self.assertEqual(dp.ca_df.iloc[0]["value"], 2.0)


if __name__ == "__main__":
    unittest.main()
