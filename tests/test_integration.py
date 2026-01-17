import time
import unittest
from datetime import datetime
from typing import Any, Dict

import numpy as np

from src.core.data_platform import DataPlatform
from src.core.execution_handler import ExecutionHandler, OrderState
from src.core.portfolio_manager import PortfolioManager
from src.gateways.base import DataProvider, ExecutionBackend


class MockPlugin(DataProvider, ExecutionBackend):
    """Combines Data and Execution for integration testing."""

    def __init__(self) -> None:
        self.positions = {"AAPL": 0.0, "MSFT": 0.0, "GOOG": 0.0}
        self.prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 2800.0}

    def fetch_bars(
        self, tickers: list[str], start: datetime, end: datetime
    ) -> Any:
        import pandas as pd

        data = []
        for t in tickers:
            data.append(
                {
                    "ticker": t,
                    "timestamp": start,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000,
                }
            )
        return pd.DataFrame(data)

    def fetch_corporate_actions(self, *args: Any, **kwargs: Any) -> Any:
        import pandas as pd

        return pd.DataFrame()

    def fetch_events(self, *args: Any, **kwargs: Any) -> Any:
        import pandas as pd

        return pd.DataFrame()

    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        qty = quantity if side == "BUY" else -quantity
        self.positions[ticker] = self.positions.get(ticker, 0.0) + qty
        return True

    def get_positions(self) -> Dict[str, float]:
        return self.positions

    def get_prices(self, tickers: list[str]) -> Dict[str, float]:
        return {t: self.prices.get(t, 100.0) for t in tickers}


class TestFullSystemIntegration(unittest.TestCase):
    def test_initialization(self) -> None:
        plugin = MockPlugin()
        _ = DataPlatform(provider=plugin, clear=True)
        _ = ExecutionHandler(backend=plugin)
        _ = PortfolioManager()

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

        # Verify positions changed from zero
        pos = plugin.get_positions()
        self.assertNotEqual(sum(abs(v) for v in pos.values()), 0.0)


if __name__ == "__main__":
    unittest.main()
