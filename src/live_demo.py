import random
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from src.alpha_library.models import MomentumModel
from src.backtesting.demo import MarketDataMock
from src.core.alpha_engine import (
    AlphaEngine,
    ModelRunConfig,
    SignalCombiner,
    SignalProcessor,
)
from src.core.data_platform import DataPlatform
from src.core.execution_handler import ExecutionHandler
from src.core.portfolio_manager import PortfolioManager
from src.core.types import Bar, QueryConfig, Timeframe
from src.gateways.base import ExecutionBackend


class MockAlpaca(ExecutionBackend):
    """Simulates Alpaca's behavior and real-time data stream."""

    def __init__(self, tickers: List[str]) -> None:
        self.tickers = tickers
        self.positions: Dict[str, float] = {t: 0.0 for t in tickers}
        self.prices: Dict[str, float] = {t: 100.0 for t in tickers}

    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        qty = quantity if side.upper() == "BUY" else -quantity
        self.positions[ticker] += qty
        print(
            f"[MOCK ALPACA] Filled {side} {quantity:.2f} {ticker} @ "
            f"{self.prices[ticker]:.2f}"
        )
        return True

    def get_positions(self) -> Dict[str, float]:
        return self.positions

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: self.prices[t] for t in tickers}

    def simulate_next_bars(self, timestamp: datetime) -> List[Bar]:
        """Generates random 1min bars."""
        bars = []
        for ticker in self.tickers:
            old_price = self.prices[ticker]
            change = random.uniform(-0.001, 0.001)
            new_price = old_price * (1 + change)
            self.prices[ticker] = new_price
            bars.append(
                Bar(
                    internal_id=0,  # Demo will map this
                    timestamp=timestamp,
                    open=old_price,
                    high=max(old_price, new_price),
                    low=min(old_price, new_price),
                    close=new_price,
                    volume=100,
                    timeframe=Timeframe.MINUTE,
                    timestamp_knowledge=timestamp,  # Simulating knowledge time
                )
            )
        return bars


def run_live_demo() -> None:
    print("\n" + "=" * 50)
    print("RUNNING STANDALONE LIVE TRADING SIMULATION")
    print("=" * 50)
    tickers = ["AAPL", "MSFT"]

    # 1. Setup
    # Use MarketDataMock for pre-populating history
    data = DataPlatform(
        db_path="./.arctic_live_demo_db",
        clear=True,
        provider=MarketDataMock(),
    )
    pm = PortfolioManager()
    mock_alpaca = MockAlpaca(tickers)
    executor = ExecutionHandler(mock_alpaca, data)

    # Pre-populate history for models using 1min data (lowest granularity)
    start_hist = datetime(2026, 1, 1, 0, 0)
    end_hist = datetime(2026, 1, 1, 9, 0)
    data.sync_data(tickers, start_hist, end_hist, timeframe=Timeframe.MINUTE)

    # 2. Live Simulation Loop
    current_time = datetime(2026, 1, 1, 9, 30)
    ticker_to_iid = {t: data.get_internal_id(t) for t in tickers}
    iids = list(ticker_to_iid.values())
    models = [MomentumModel()]

    # A. Daily Risk Model & Factor Return Estimation (Once per day)
    # Query 1min data and resample to 30min for the risk model
    hist_bars = data.get_bars(
        iids,
        QueryConfig(
            start=current_time - timedelta(days=5),
            end=current_time - timedelta(minutes=1),
            timeframe=Timeframe.MIN_30,
        ),
    )
    expected_factor_returns = None
    if not hist_bars.empty:
        pivot_rets = (
            hist_bars.pivot(
                index="timestamp",
                columns="internal_id",
                values="close_30min",
            )
            .pct_change()
            .dropna()
        )
        if not pivot_rets.empty:
            pm.update_risk_model(pivot_rets.values)
            hist_f_rets = pm.get_factor_returns(pivot_rets.values)
            expected_factor_returns = np.mean(hist_f_rets, axis=0)

    print(f"Trading started at {current_time}...")

    for i in range(60):  # Simulate 60 minutes
        step_time = current_time + timedelta(minutes=i)
        bars = mock_alpaca.simulate_next_bars(step_time)

        # ExecutionHandler is the ONLY component writing to DataPlatform
        # It writes 1min bars directly
        for b in bars:
            ticker = tickers[bars.index(b)]
            b.internal_id = ticker_to_iid[ticker]
            executor.on_bar(b)

        # Every 30 minutes, check for rebalancing
        if (i + 1) % 30 == 0:
            print(
                f"\n[ORCHESTRATOR] Step {i + 1} | Rebalancing at {step_time}"
            )

            # B. Generate Signals
            signals = [
                AlphaEngine.run_model(
                    data, m, iids, ModelRunConfig(step_time, Timeframe.MIN_30)
                )
                for m in models
            ]
            combined = SignalCombiner.combine(
                [SignalProcessor.zscore(s) for s in signals]
            )

            # C. Optimization using cached Daily Risk Model and Factor Drift
            pm.optimize(combined, factor_returns=expected_factor_returns)

            # D. Goal Position Calculation & Rebalance
            prices = mock_alpaca.get_prices(tickers)
            reverse_ism = data.reverse_ism
            goal_positions = {
                reverse_ism[iid]: (weight * 100_000.0)
                / prices[reverse_ism[iid]]
                for iid, weight in pm.current_weights.items()
                if reverse_ism[iid] in prices and prices[reverse_ism[iid]] > 0
            }
            executor.rebalance(goal_positions, interval=0)

    print("\nSimulation finished.")
    print(f"Final Positions: {mock_alpaca.get_positions()}")


if __name__ == "__main__":
    import src.alpha_library.features  # noqa: F401

    run_live_demo()
