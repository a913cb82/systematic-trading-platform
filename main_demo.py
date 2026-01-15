from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.alpha_engine import AlphaModel
from src.base import DataProvider, ExecutionBackend
from src.data_platform import DataPlatform
from src.execution_handler import ExecutionHandler
from src.portfolio_manager import PortfolioManager


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
        print(f"ORDER: {side} {quantity} shares of {ticker}")
        return True

    def get_positions(self) -> Dict[str, float]:
        return {}

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: 100.0 for t in tickers}


class MeanReversion(AlphaModel):
    def compute_signals(
        self, latest_features: pd.DataFrame, returns: pd.DataFrame
    ) -> Dict[int, float]:
        # Target negative of 1-day returns
        if returns.empty:
            return {}
        latest_rets = returns.iloc[-1]
        return {iid: -float(latest_rets[iid]) for iid in latest_features.index}


def main() -> None:
    # 1. Platform Setup
    plugin = MockPlugin()
    data = DataPlatform(provider=plugin)
    pm = PortfolioManager(tc_penalty=0.01)
    exec_h = ExecutionHandler(backend=plugin)

    # 2. Sync and Map
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]
    data.sync_data(tickers, datetime(2025, 1, 1), datetime(2025, 1, 15))
    iids = [data.get_internal_id(t) for t in tickers]

    # 3. Generate Signals
    model = MeanReversion(data, features=["sma_10", "rsi_14"])
    forecasts = model.generate_forecasts(iids, datetime(2025, 1, 15))

    # 4. Optimize
    hist_returns = np.random.randn(30, 4) * 0.01
    weights = pm.optimize(forecasts, hist_returns)

    print("\n--- Optimized Portfolio Weights ---")
    for iid, w in weights.items():
        print(f"{data.reverse_ism[iid]}: {w:.2%}")

    # Execute
    min_weight = 0.01
    for iid, w in weights.items():
        if abs(w) > min_weight:
            side = "BUY" if w > 0 else "SELL"
            exec_h.execute_direct(data.reverse_ism[iid], abs(w) * 100, side)


if __name__ == "__main__":
    main()
