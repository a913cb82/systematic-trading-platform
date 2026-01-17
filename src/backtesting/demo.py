from datetime import datetime, timedelta
from typing import List

import pandas as pd

from src.alpha_library.models import (
    EarningsModel,
    MomentumModel,
    ReversionModel,
)
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.data_platform import DataPlatform
from src.core.portfolio_manager import PortfolioManager
from src.gateways.base import DataProvider


class MarketDataMock(DataProvider):
    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="30min")
        data = []
        for ticker in tickers:
            for timestamp in dates:
                price = 100.0 + (pd.Timestamp(timestamp).hour - 10) * 0.1
                data.append(
                    {
                        "ticker": ticker,
                        "timestamp": timestamp,
                        "open": price - 0.1,
                        "high": price + 0.5,
                        "low": price - 0.4,
                        "close": price,
                        "volume": 50000.0,
                        "timeframe": "30min",
                    }
                )
        return pd.DataFrame(data)

    def fetch_corporate_actions(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "ex_date", "type", "ratio"])

    def fetch_events(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        data = []
        target_dt = pd.Timestamp(start).normalize() + timedelta(
            days=1, hours=16
        )
        if "AAPL" in tickers and start <= target_dt <= end:
            data.append(
                {
                    "ticker": "AAPL",
                    "timestamp": target_dt,
                    "event_type": "EARNINGS_RELEASE",
                    "value": {"surprise_pct": 0.05},
                }
            )
        return pd.DataFrame(data)


def main() -> None:
    # 1. Setup Data & PM
    mock = MarketDataMock()
    data = DataPlatform(provider=mock, clear=True)
    pm = PortfolioManager(max_pos=0.15, risk_aversion=1.5)

    # 2. Sync Data
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 5)
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "META"]
    data.sync_data(tickers, start, end)

    # 3. Engine Initialization
    engine = BacktestEngine(data, pm)

    # 4. Run Multi-Model Backtest
    print(f"Starting Backtest: {start.date()} to {end.date()}...")
    stats = engine.run(
        BacktestConfig(
            start_date=start,
            end_date=end,
            alpha_models=[
                MomentumModel(),
                ReversionModel(),
                EarningsModel(),
            ],
            weights=[0.4, 0.4, 0.2],
            tickers=tickers,
        )
    )

    # 5. Output Results
    print("\n--- Backtest Report ---")
    print(f"Status: {stats['status']}")
    print(f"Total Return: {stats['total_return']:.2%}")
    print(f"Sharpe Ratio: {stats['sharpe']:.2f}")
    if "drawdown" in stats and stats["drawdown"]:
        print(f"Max Drawdown: {stats['drawdown']['max_dd']:.2%}")
    print(f"Final Equity: ${stats['final_equity']:,.2f}")

    print("\n--- Periodic Performance ---")
    print(stats["performance_table"])


if __name__ == "__main__":
    main()
