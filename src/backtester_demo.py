from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

import src.alpha_library.features  # noqa: F401
from src.alpha_library.models import MomentumModel, ValueModel
from src.core.alpha_engine import SignalCombiner, SignalProcessor
from src.core.data_platform import DataPlatform
from src.core.execution_handler import ExecutionHandler, TCAEngine
from src.core.portfolio_manager import PortfolioManager
from src.gateways.base import DataProvider, ExecutionBackend


class IntraDayInstitutionalPlugin(DataProvider, ExecutionBackend):
    """
    Simulates a high-fidelity intra-day provider.
    """

    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        # Use 30-minute frequency for intra-day simulation
        dates = pd.date_range(start, end, freq="30min")
        data = []
        for t in tickers:
            # Seed based on ticker name for deterministic prices
            np.random.seed(sum(ord(c) for c in t))
            price = 100.0 + np.random.randn() * 5.0
            for d in dates:
                # Random walk
                price += np.random.randn() * 0.5
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
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "ex_date", "type", "ratio"])

    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        # Silently process orders for the loop
        return True

    def get_positions(self) -> Dict[str, float]:
        # Track positions if needed, using rebalance logic for now
        return {}

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: 100.0 for t in tickers}


def run_hedge_fund_simulation() -> None:
    print("=== Institutional Hedge Fund: Full Trading Day Simulation ===\n")

    # 1. Infrastructure Setup
    plugin = IntraDayInstitutionalPlugin()
    data = DataPlatform(provider=plugin)
    pm = PortfolioManager(
        risk_aversion=2.0,
        tc_penalty=0.001,
        leverage_limit=1.0,
    )
    exec_h = ExecutionHandler(backend=plugin)

    # 2. Data Ingestion (History + Today)
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA"]
    sim_day = datetime(2026, 1, 16)
    history_start = sim_day - timedelta(days=5)  # 5 days of 30m bars

    print(f"Syncing market data from {history_start} to market close...")
    data.sync_data(tickers, history_start, sim_day + timedelta(hours=16))
    iids = [data.get_internal_id(t) for t in tickers]
    reverse_ism = data.reverse_ism

    # 3. Risk Model Calculation (Once at start of day)
    print("Calculating PCA Factor Risk Model for the day...")
    # Simulate historical returns (60 observations x 5 assets)
    returns_history = np.random.randn(60, len(iids)) * 0.01
    pm.update_risk_model(returns_history)

    # 4. Intra-day Simulation Loop (09:30 to 16:00)
    trading_times = pd.date_range(
        sim_day + timedelta(hours=9, minutes=30),
        sim_day + timedelta(hours=16),
        freq="30min",
    )

    print(f"Starting execution loop over {len(trading_times)} intervals.\n")
    print(
        f"{'Timestamp':<17} | {'Net Exp':<10} | {'Gross Lev':<10} | Top Trade"
    )
    print("-" * 75)

    aum = 10_000_000.0

    for ts in trading_times:
        # A. Alpha Generation
        mom_model = MomentumModel(data, features=["rsi_14"])
        val_model = ValueModel(data, features=["sma_10", "close"])

        sig_mom = mom_model.generate_forecasts(iids, ts)
        sig_val = val_model.generate_forecasts(iids, ts)

        # B. Combination
        combined = SignalCombiner.combine(
            [
                SignalProcessor.zscore(sig_mom),
                SignalProcessor.zscore(sig_val),
            ],
            weights=[0.5, 0.5],
        )

        # C. Optimization (Uses cached PCA Risk Model)
        weights = pm.optimize(combined)

        # D. Execution (Rebalance)
        exec_h.rebalance(weights, reverse_ism, aum)

        # E. Reporting
        net_exp = sum(weights.values())
        gross_lev = sum(abs(w) for w in weights.values())

        top_iid = max(weights, key=lambda i: abs(weights[i]))
        top_ticker = reverse_ism[top_iid]
        top_val = weights[top_iid]

        ts_str = ts.strftime("%Y-%m-%d %H:%M")
        print(
            f"{ts_str:<17} | {net_exp:>10.4f} | {gross_lev:>10.2f} | "
            f"{top_ticker}: {top_val:>7.2%}"
        )

    print("\n--- Final Day Summary ---")
    print(f"Total Orders Managed: {len(exec_h.orders)}")
    filled = [o for o in exec_h.orders if o.state.value == "FILLED"]
    print(f"Total Slices Filled:  {len(filled)} (via Algorithmic Engine)")

    # TCA Example
    last_order = exec_h.orders[-1]
    slippage = TCAEngine.calculate_slippage(100.0, 100.02, last_order.side)
    print(
        f"Final Execution TCA:  {slippage:.1f} bps slippage on "
        f"{last_order.ticker}"
    )

    print("\n=== Institutional Simulation Complete ===")


if __name__ == "__main__":
    run_hedge_fund_simulation()
