from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.alpha_library.models import (
    ResidualMomentumModel,
    ResidualReversionModel,
)
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.data_platform import DataPlatform
from src.core.portfolio_manager import PortfolioManager
from src.gateways.base import DataProvider, ExecutionBackend


class MarketDataMock(DataProvider, ExecutionBackend):
    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq="30min")
        data = []
        for t in tickers:
            for d in dates:
                price = 100.0 + (pd.Timestamp(d).hour - 10) * 0.1
                data.append(
                    {
                        "ticker": t,
                        "timestamp": d,
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

    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        return True

    def get_positions(self) -> Dict[str, float]:
        return {}

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: 100.0 for t in tickers}


def main() -> None:
    # 1. Setup Data & PM
    mock = MarketDataMock()
    data = DataPlatform(provider=mock)
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
            alpha_models=[ResidualMomentumModel(), ResidualReversionModel()],
            weights=[0.5, 0.5],
            tickers=tickers,
        )
    )

    # 5. Output Results
    print("\n--- Backtest Report ---")
    print(f"Status: {stats['status']}")
    print(f"Total Return: {stats['total_return']:.2%}")
    print(f"Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"Max Drawdown: {stats['drawdown']:.2%}")
    print(f"Final Equity: ${stats['final_equity']:,.2f}")

    print("\n--- Attribution ---")
    attr = stats["attribution"]
    print(f"Factor PnL: ${attr['factor']:,.2f}")
    print(f"Idiosyncratic PnL: ${attr['selection']:,.2f}")


if __name__ == "__main__":
    main()
