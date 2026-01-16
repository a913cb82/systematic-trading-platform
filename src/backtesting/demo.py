from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

import src.alpha_library.features  # noqa: F401
from src.alpha_library.models import (
    ResidualMomentumModel,
    ResidualReversionModel,
)
from src.backtesting.engine import BacktestEngine
from src.core.data_platform import DataPlatform, Event
from src.core.portfolio_manager import PortfolioManager
from src.gateways.base import DataProvider, ExecutionBackend


class MarketDataMock(DataProvider, ExecutionBackend):
    """
    Synthetic data provider.
    """

    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="30min")
        data = []
        for t in tickers:
            np.random.seed(sum(ord(c) for c in t))
            price = 100.0 + np.random.randn() * 5.0
            for d in dates:
                price += np.random.randn() * 0.4
                data.append(
                    {
                        "ticker": t,
                        "timestamp": d,
                        "open": price - 0.2,
                        "high": price + 0.4,
                        "low": price - 0.4,
                        "close": price,
                        "volume": 50000.0,
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


def run_demo() -> None:
    print("=== Backtest Simulation: Idiosyncratic Alpha ===\n")

    # 1. Setup
    plugin = MarketDataMock()
    data = DataPlatform(provider=plugin)
    # Tight safety limit to exercise logic
    pm = PortfolioManager(
        risk_aversion=2.0, tc_penalty=0.001, max_drawdown=-0.2
    )
    engine = BacktestEngine(data, pm)

    # 2. Ingest Data
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA"]
    start_date = datetime(2026, 1, 12)
    end_date = datetime(2026, 1, 16)

    print(
        f"Ingesting {len(tickers)} assets from {start_date.date()} "
        f"to {end_date.date()}..."
    )
    data.sync_data(
        tickers,
        start_date - timedelta(days=10),
        end_date + timedelta(hours=16),
    )

    # Exercise DataPlatform: add_events and get_events
    print("Hydrating Event Store...")
    for t in tickers:
        data.add_events(
            [Event(data.get_internal_id(t), start_date, "REBAL_SIGNAL", 1.0)]
        )

    # 3. Models (Targeting residuals)
    alpha_models = [
        ResidualMomentumModel(data, features=["residual_mom_10"]),
        ResidualReversionModel(
            data, features=["returns_residual", "residual_vol_20"]
        ),
    ]

    # 4. Run (Using dynamic universe from PIT Security Master)
    stats = engine.run(
        start_date=start_date,
        end_date=end_date,
        alpha_models=alpha_models,
        weights=[0.5, 0.5],
    )

    # 5. Report
    print("\n" + "=" * 40)
    print("PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"Status:       {stats['status']}")
    print(f"Total Return: {stats['total_return']:.2%}")
    print(f"Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"Max Drawdown: {stats['drawdown']['max_dd']:.2%}")
    print(f"Final Equity: ${stats['final_equity']:,.2f}")
    print("-" * 40)
    print("PnL ATTRIBUTION")
    attr = stats["attribution"]
    print(f"Factor PnL:    ${attr['factor']:,.2f}")
    print(f"Selection PnL: ${attr['selection']:,.2f}")

    # Exercise reverse_ism and get_universe
    print("\nFinal Universe Map:")
    for iid in data.get_universe(end_date):
        print(f"  {iid} -> {data.reverse_ism[iid]}")
    print("=" * 40)


if __name__ == "__main__":
    run_demo()
