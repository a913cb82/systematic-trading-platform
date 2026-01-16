from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.backtesting.analytics import PerformanceAnalyzer
from src.core.alpha_engine import SignalCombiner, SignalProcessor
from src.core.data_platform import DataPlatform
from src.core.portfolio_manager import PortfolioManager


class BacktestEngine:
    """
    Manages multi-day trading simulations with risk and safety enforcement.
    """

    def __init__(
        self,
        data: DataPlatform,
        pm: PortfolioManager,
        initial_capital: float = 10_000_000.0,
    ):
        self.data = data
        self.pm = pm
        self.capital = initial_capital
        self.equity_curve = [initial_capital]
        self.daily_returns: List[float] = []
        self.total_attribution = {"factor": 0.0, "selection": 0.0}
        self.status = "ACTIVE"

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        alpha_models: List,
        weights: List[float],
        tickers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Runs the simulation. If tickers is None, uses the dynamic PIT universe.
        """
        trading_days = pd.date_range(start_date, end_date, freq="B")

        for day in trading_days:
            if self.status == "KILLED":
                break

            # 1. Dynamic Universe Selection
            if tickers is not None:
                iids = sorted([self.data.get_internal_id(t) for t in tickers])
            else:
                iids = sorted(self.data.get_universe(day))

            if not iids:
                continue

            # 2. Risk Model Update
            hist_df = self.data.get_bars(
                iids, day - timedelta(days=5), day - timedelta(minutes=1)
            )
            if hist_df.empty:
                continue

            pivot_rets = (
                hist_df.pivot(
                    index="timestamp", columns="internal_id", values="close"
                )
                .pct_change(fill_method=None)
                .dropna()
            )
            self.pm.update_risk_model(pivot_rets.values)

            # 3. Rebalance Cycles
            day_pnl = 0.0
            intervals = [day + timedelta(hours=10), day + timedelta(hours=15)]

            for ts in intervals:
                # Safety Check (Kill Switch / Rate Limiting)
                current_equity = self.equity_curve[-1] + day_pnl
                if not self.pm.check_safety(current_equity / self.capital):
                    self.status = "KILLED"
                    print(f"!!! STOP: Risk Kill-Switch triggered at {ts} !!!")
                    break

                # Alpha Generation
                from src.core.alpha_engine import AlphaEngine

                signals = []
                for model in alpha_models:
                    signals.append(
                        AlphaEngine.run_model(self.data, model, iids, ts)
                    )

                combined = SignalCombiner.combine(
                    [SignalProcessor.zscore(s) for s in signals],
                    weights=weights,
                )

                weights_opt = self.pm.optimize(combined)

                # 4. Simulate Returns
                current_bars = self.data.get_bars(iids, ts, ts)
                next_ts = ts + timedelta(hours=1)
                next_bars = self.data.get_bars(iids, next_ts, next_ts)

                if not current_bars.empty and not next_bars.empty:
                    p0 = {
                        row["internal_id"]: float(row["close"])
                        for _, row in current_bars.iterrows()
                    }
                    p1 = {
                        row["internal_id"]: float(row["close"])
                        for _, row in next_bars.iterrows()
                    }

                    # Calculate realized returns only for assets in both bars
                    common_iids = set(p0.keys()) & set(p1.keys())
                    rets = {i: (p1[i] / p0[i] - 1) for i in common_iids}

                    if self.pm.loadings is not None:
                        # Attribution uses weights only for assets
                        # we have returns for
                        active_weights = {
                            i: weights_opt.get(i, 0.0) for i in common_iids
                        }
                        attr = PerformanceAnalyzer.factor_attribution(
                            active_weights, rets, self.pm.loadings
                        )
                        self.total_attribution["factor"] += (
                            attr["factor"] * current_equity
                        )
                        self.total_attribution["selection"] += (
                            attr["selection"] * current_equity
                        )
                        day_pnl += attr["total"] * current_equity

            # Record Daily Stats
            daily_ret = day_pnl / self.equity_curve[-1]
            self.daily_returns.append(daily_ret)
            self.equity_curve.append(self.equity_curve[-1] + day_pnl)

        return self.report()

    def report(self) -> Dict:
        ret_series = pd.Series(self.daily_returns)
        equity_series = pd.Series(self.equity_curve)

        return {
            "status": self.status,
            "total_return": self.equity_curve[-1] / self.equity_curve[0] - 1,
            "sharpe": PerformanceAnalyzer.calculate_sharpe(ret_series),
            "drawdown": PerformanceAnalyzer.calculate_drawdown(equity_series),
            "final_equity": self.equity_curve[-1],
            "attribution": self.total_attribution,
        }
