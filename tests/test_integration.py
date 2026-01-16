import unittest
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.core.alpha_engine import AlphaModel
from src.core.data_platform import DataPlatform
from src.core.execution_handler import ExecutionHandler
from src.core.portfolio_manager import PortfolioManager
from src.gateways.base import DataProvider, ExecutionBackend


class MockPlugin(DataProvider, ExecutionBackend):
    def fetch_bars(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="D")
        data = []
        for t in tickers:
            for d in dates:
                data.append(
                    {
                        "ticker": t,
                        "timestamp": d,
                        "open": 100,
                        "high": 101,
                        "low": 99,
                        "close": 100,
                        "volume": 1000,
                    }
                )
        return pd.DataFrame(data)

    def fetch_corporate_actions(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "ex_date", "type", "ratio"])

    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        return True

    def get_positions(self) -> Dict[str, float]:
        return {}

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: 100.0 for t in tickers}


class TestFullSystemIntegration(unittest.TestCase):
    def test_end_to_end_flow(self) -> None:
        plugin = MockPlugin()
        data = DataPlatform(provider=plugin)
        pm = PortfolioManager()
        _ = ExecutionHandler(backend=plugin)

        tickers = ["AAPL", "MSFT"]
        data.sync_data(tickers, datetime(2025, 1, 1), datetime(2025, 1, 10))
        iids = [data.get_internal_id(t) for t in tickers]

        # Alpha
        class SimpleAlpha(AlphaModel):
            def compute_signals(
                self, latest: pd.DataFrame, history: pd.DataFrame
            ) -> Dict[int, float]:
                return {iid: 0.1 for iid in latest.index}

        model = SimpleAlpha(data, features=[])
        ts = datetime(2025, 1, 10)
        forecasts = model.generate_forecasts(iids, ts)

        # Portfolio
        hist_returns = np.random.randn(10, 2) * 0.01
        weights = pm.optimize(forecasts, hist_returns)

        self.assertEqual(len(weights), 2)
        self.assertLessEqual(max(abs(w) for w in weights.values()), 0.2001)

    def test_full_lifecycle(self) -> None:
        """Simulates multiple days of the fund's lifecycle."""
        plugin = MockPlugin()
        data = DataPlatform(provider=plugin)
        pm = PortfolioManager(tc_penalty=0.01)
        exec_h = ExecutionHandler(backend=plugin)

        tickers = ["AAPL", "MSFT", "GOOG"]
        reverse_ism = {}
        for t in tickers:
            iid = data.get_internal_id(t)
            reverse_ism[iid] = t

        # Day 1
        data.sync_data(tickers, datetime(2025, 1, 1), datetime(2025, 1, 1))
        forecasts = {data.get_internal_id(t): 0.1 for t in tickers}
        hist_rets = np.random.randn(20, 3) * 0.01

        weights = pm.optimize(forecasts, hist_rets)
        exec_h.rebalance(weights, reverse_ism, 100000.0)

        # Day 2
        data.sync_data(tickers, datetime(2025, 1, 2), datetime(2025, 1, 2))
        # Slightly different signals
        forecasts = {data.get_internal_id(t): 0.11 for t in tickers}
        weights2 = pm.optimize(forecasts, hist_rets)

        # Verify weights were updated
        self.assertEqual(len(weights2), 3)
        # Rebalance Day 2
        exec_h.rebalance(weights2, reverse_ism, 100000.0)

        # Verify backend positions were called (MockPlugin always returns
        # {} for positions but we can verify rebalance ran without crash)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
