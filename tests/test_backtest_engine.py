import unittest
from datetime import datetime
from typing import List

import pandas as pd

from src.alpha_library.models import MomentumModel
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.data_platform import DataPlatform
from src.core.portfolio_manager import PortfolioManager
from src.core.types import Timeframe
from src.gateways.base import (
    BarProvider,
    CorporateActionProvider,
    EventProvider,
)


class MockProvider(BarProvider, CorporateActionProvider, EventProvider):
    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = Timeframe.DAY,
    ) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="1h")
        data = []
        for ticker in tickers:
            for ts in dates:
                data.append(
                    {
                        "ticker": ticker,
                        "timestamp": ts,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                        "volume": 1000,
                        "timeframe": timeframe,
                    }
                )
        return pd.DataFrame(data)

    def fetch_corporate_actions(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "ex_date", "type", "value"])

    def fetch_events(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["ticker", "timestamp", "event_type", "value"]
        )


class TestBacktestEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MockProvider()
        self.data = DataPlatform(
            self.provider, db_path="./.arctic_test_db", clear=True
        )
        self.pm = PortfolioManager()
        self.engine = BacktestEngine(self.data, self.pm)

    def test_engine_run_flow(self) -> None:
        """
        Verify that the engine can run a multi-day simulation
        and generate a valid report.
        """
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 3)
        tickers = ["AAPL", "MSFT"]

        # Ensure data is present
        self.data.sync_data(tickers, start, end, timeframe=Timeframe.MIN_30)

        config = BacktestConfig(
            start_date=start,
            end_date=end,
            alpha_models=[MomentumModel()],
            weights=[1.0],
            tickers=tickers,
            timeframe=Timeframe.MIN_30,
        )

        report = self.engine.run(config)

        self.assertEqual(report["status"], "ACTIVE")
        self.assertIn("total_return", report)
        self.assertIn("sharpe", report)
        self.assertTrue(len(self.engine.interval_results) > 0)


if __name__ == "__main__":
    import src.alpha_library.features  # noqa: F401

    unittest.main()
