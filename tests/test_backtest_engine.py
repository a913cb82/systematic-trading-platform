import unittest
from datetime import datetime
from typing import List

import pandas as pd

from src.alpha_library.models import ResidualMomentumModel
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.data_platform import DataPlatform
from src.core.portfolio_manager import PortfolioManager
from src.gateways.base import DataProvider


class MockProvider(DataProvider):
    def fetch_bars(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="30min")
        data = []
        for t in tickers:
            for d in dates:
                data.append(
                    {
                        "ticker": t,
                        "timestamp": d,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                        "volume": 1000.0,
                        "timeframe": "30min",
                    }
                )
        return pd.DataFrame(data)

    def fetch_corporate_actions(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "ex_date", "type", "ratio"])


class TestBacktestEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "./.arctic_test_db"
        self.provider = MockProvider()
        self.data = DataPlatform(
            provider=self.provider, db_path=self.db_path, clear=True
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
        self.data.sync_data(tickers, start, end)

        config = BacktestConfig(
            start_date=start,
            end_date=end,
            alpha_models=[ResidualMomentumModel()],
            weights=[1.0],
            tickers=tickers,
            timeframe="30min",
        )

        report = self.engine.run(config)

        self.assertEqual(report["status"], "ACTIVE")
        self.assertTrue("total_return" in report)
        self.assertTrue("sharpe" in report)
        self.assertGreater(len(self.engine.equity_curve), 1)
        self.assertFalse(report["performance_table"].empty)


if __name__ == "__main__":
    unittest.main()
