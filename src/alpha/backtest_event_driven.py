from datetime import datetime
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd

from ..common.base import AlphaModel, MarketDataProvider, PortfolioOptimizer
from ..common.types import Trade


class EventDrivenBacktester:
    def __init__(  # noqa: PLR0913
        self,
        market_data: MarketDataProvider,
        alpha_model: AlphaModel,
        optimizer: PortfolioOptimizer,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1_000_000.0,
        transaction_cost_bps: float = 5.0,
    ) -> None:
        self.market_data = market_data
        self.alpha_model = alpha_model
        self.optimizer = optimizer
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.tc_multiplier = transaction_cost_bps / 10000.0

        self.cash = initial_capital
        self.positions: Dict[int, float] = {}  # internal_id -> shares

        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Trade] = []
        self.current_prices: Dict[int, float] = {}

    def run(self) -> None:
        universe = self.market_data.get_universe(self.start_date)
        # Fetch all bars for the period
        bars = self.market_data.get_bars(
            universe, self.start_date, self.end_date, adjustment="RAW"
        )

        if not bars:
            return

        df_bars = pd.DataFrame(bars)
        df_bars["timestamp"] = pd.to_datetime(df_bars["timestamp"])

        # Group bars by timestamp to simulate discrete time steps
        for ts, group in df_bars.groupby("timestamp"):
            self._step(cast(datetime, ts), group)

    def _step(self, timestamp: datetime, current_bars: pd.DataFrame) -> None:
        # 1. Update current prices (as of this timestamp)
        for _, row in current_bars.iterrows():
            self.current_prices[row["internal_id"]] = row["close"]

        # 2. Mark to Market
        total_equity = self.cash
        for iid, shares in self.positions.items():
            total_equity += shares * self.current_prices.get(iid, 0.0)

        # 3. Generate Forecasts
        # IMPORTANT: In a real system, we must ensure alpha_model only uses data  # noqa: E501
        # that was known BEFORE this trade price was established.
        # For this PoC, we assume the model generates signals at T close
        # and we execute at T close (slight look-ahead if not careful).
        forecasts = self.alpha_model.generate_forecasts(timestamp)

        # 4. Optimize to get Target Weights
        target_weights = self.optimizer.optimize(timestamp, forecasts)

        # 5. Simulate Execution
        self._execute_trades(timestamp, target_weights, total_equity)

        # 6. Record state
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": total_equity,
                "cash": self.cash,
                "positions": self.positions.copy(),
            }
        )

    def _execute_trades(
        self,
        timestamp: datetime,
        target_weights: Dict[int, float],
        total_equity: float,
    ) -> None:
        # Target weight -> Target value -> Target shares
        target_shares = {}
        for iid, weight in target_weights.items():
            price = self.current_prices.get(iid)
            if price and price > 0:
                target_shares[iid] = (total_equity * weight) / price
            else:
                target_shares[iid] = 0.0

        # Calculate diffs
        all_ids = set(self.positions.keys()) | set(target_shares.keys())

        for iid in all_ids:
            current_s = self.positions.get(iid, 0.0)
            target_s = target_shares.get(iid, 0.0)
            diff = target_s - current_s

            if diff == 0:
                continue

            price = self.current_prices.get(iid)
            if not price:
                continue

            # Execute
            cost = diff * price
            tc = abs(cost) * self.tc_multiplier

            self.cash -= cost + tc
            self.positions[iid] = target_s

            self.trades.append(
                {
                    "internal_id": iid,
                    "timestamp": timestamp,
                    "side": "BUY" if diff > 0 else "SELL",
                    "quantity": abs(diff),
                    "price": price,
                    "fees": tc,
                    "venue": "BACKTEST",
                }
            )

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def calculate_metrics(self) -> Dict[str, float]:
        df = self.get_results()
        min_required_bars = 2
        if df.empty or len(df) < min_required_bars:
            return {}

        df = pd.DataFrame(self.equity_curve)
        df["returns"] = df["equity"].pct_change(fill_method=None).fillna(0)

        sharpe = 0.0

        daily_pnl = df["returns"]
        std = daily_pnl.std()
        sharpe = np.sqrt(252) * daily_pnl.mean() / std if std != 0 else 0
        cumulative = (df["equity"].iloc[-1] / self.initial_capital) - 1

        # Max Drawdown
        cum_ret = df["equity"]
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()

        return {
            "sharpe": sharpe,
            "cumulative_return": cumulative,
            "max_drawdown": max_dd,
            "final_equity": df["equity"].iloc[-1],
            "total_trades": len(self.trades),
        }
