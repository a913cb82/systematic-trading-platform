import time
import unittest
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.core.data_platform import DataPlatform
from src.core.execution_handler import ExecutionHandler
from src.core.portfolio_manager import PortfolioManager
from src.core.types import OrderState, Timeframe
from src.gateways.base import DataProvider, ExecutionBackend


class MockPlugin(DataProvider, ExecutionBackend):
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
                    "ticker": t,
                    "timestamp": start,
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "volume": 1000,
                    "timeframe": timeframe,
                }
                for t in tickers
            ]
        )

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

    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        return True

    def get_positions(self) -> Dict[str, float]:
        return {}

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: 100.0 for t in tickers}


class TestFullSystemIntegration(unittest.TestCase):
    def test_full_lifecycle(self) -> None:
        """Simulates multiple days of the fund's lifecycle."""
        plugin = MockPlugin()
        data = DataPlatform(provider=plugin, clear=True)
        pm = PortfolioManager(tc_penalty=0.01)
        exec_h = ExecutionHandler(backend=plugin)

        tickers = ["AAPL", "MSFT", "GOOG"]
        reverse_ism = {}
        for t in tickers:
            iid = data.get_internal_id(t)
            reverse_ism[iid] = t

        # Day 1
        data.sync_data(tickers, datetime(2025, 1, 1), datetime(2025, 1, 1))
        # Stronger forecast to ensure target positions exceed tolerance
        forecasts = {data.get_internal_id(t): 0.5 for t in tickers}
        hist_rets = np.random.randn(20, 3) * 0.01

        weights = pm.optimize(forecasts, hist_rets)

        # Calculate goal positions locally
        prices = plugin.get_prices(tickers)
        capital = 1000000.0  # Increase capital
        goal_positions = {
            reverse_ism[iid]: (weight * capital) / prices[reverse_ism[iid]]
            for iid, weight in weights.items()
        }

        exec_h.rebalance(goal_positions, interval=0)

        # Wait for worker
        max_wait = 1.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if all(o.state == OrderState.FILLED for o in exec_h.orders):
                break
            time.sleep(0.01)

        # In this mock, get_positions doesn't reflect trades,
        # but we can check the order list.
        self.assertTrue(len(exec_h.orders) > 0)


if __name__ == "__main__":
    unittest.main()
