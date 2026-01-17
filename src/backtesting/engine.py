from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict

import pandas as pd

from src.backtesting.analytics import PerformanceAnalyzer
from src.core.alpha_engine import (
    AlphaEngine,
    AlphaModel,
    ModelRunConfig,
    SignalCombiner,
    SignalProcessor,
)
from src.core.data_platform import DataPlatform, QueryConfig
from src.core.portfolio_manager import PortfolioManager


@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    alpha_models: List[AlphaModel]
    weights: List[float]
    tickers: Optional[List[str]] = None
    timeframe: str = "30min"
    slippage_bps: float = 5.0
    market_impact_coef: float = 0.1  # bps per % of capital traded
    report_freq: str = "D"


class IntervalResult(TypedDict):
    timestamp: datetime
    gross_ret: float
    net_ret: float
    tcost: float


class BacktestReport(TypedDict):
    status: str
    total_return: float
    sharpe: float
    drawdown: Dict[str, float]
    final_equity: float
    performance_table: pd.DataFrame
    message: Optional[str]


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
        self.interval_results: List[IntervalResult] = []
        self.status = "ACTIVE"

    def _calculate_tcosts(
        self,
        weights_opt: Dict[int, float],
        prev_weights: Dict[int, float],
        config: BacktestConfig,
    ) -> float:
        total_tcost = 0.0
        all_iids = set(prev_weights.keys()) | set(weights_opt.keys())
        for iid in all_iids:
            dw = weights_opt.get(iid, 0.0) - prev_weights.get(iid, 0.0)
            if dw == 0:
                continue
            linear_cost = abs(dw) * (config.slippage_bps / 10000.0)
            impact_cost = (dw**2) * (config.market_impact_coef / 10000.0)
            total_tcost += linear_cost + impact_cost
        return total_tcost

    def _simulate_interval_returns(
        self,
        iids: List[int],
        weights_opt: Dict[int, float],
        ts: datetime,
        config: BacktestConfig,
        close_col: str,
    ) -> float:
        current_bars = self.data.get_bars(
            iids, QueryConfig(start=ts, end=ts, timeframe=config.timeframe)
        )
        next_ts = ts + timedelta(hours=1)
        next_bars = self.data.get_bars(
            iids,
            QueryConfig(
                start=next_ts, end=next_ts, timeframe=config.timeframe
            ),
        )

        interval_gross_ret = 0.0
        if not current_bars.empty and not next_bars.empty:
            p0 = {
                row["internal_id"]: float(row[close_col])
                for _, row in current_bars.iterrows()
            }
            p1 = {
                row["internal_id"]: float(row[close_col])
                for _, row in next_bars.iterrows()
            }
            common_iids = set(p0.keys()) & set(p1.keys())
            for i in common_iids:
                asset_ret = p1[i] / p0[i] - 1
                interval_gross_ret += weights_opt.get(i, 0.0) * asset_ret
        return interval_gross_ret

    def run(self, config: BacktestConfig) -> BacktestReport:
        """Runs the simulation."""
        trading_days = pd.date_range(
            config.start_date, config.end_date, freq="B"
        )

        for day in trading_days:
            if self.status == "KILLED":
                break

            iids = (
                sorted([self.data.get_internal_id(t) for t in config.tickers])
                if config.tickers
                else sorted(self.data.get_universe(day))
            )
            if not iids:
                continue

            hist_df = self.data.get_bars(
                iids,
                QueryConfig(
                    start=day - timedelta(days=5),
                    end=day - timedelta(minutes=1),
                    timeframe=config.timeframe,
                ),
            )
            if hist_df.empty:
                continue

            close_col = f"close_{config.timeframe}"
            pivot_rets = (
                hist_df.pivot(
                    index="timestamp", columns="internal_id", values=close_col
                )
                .pct_change(fill_method=None)
                .dropna()
            )
            self.pm.update_risk_model(pivot_rets.values)

            for ts in [day + timedelta(hours=10), day + timedelta(hours=15)]:
                if not self.pm.check_safety(
                    self.equity_curve[-1] / self.capital
                ):
                    self.status = "KILLED"
                    break

                signals = [
                    AlphaEngine.run_model(
                        self.data,
                        m,
                        iids,
                        ModelRunConfig(
                            timestamp=ts, timeframe=config.timeframe
                        ),
                    )
                    for m in config.alpha_models
                ]
                combined = SignalCombiner.combine(
                    [SignalProcessor.zscore(s) for s in signals],
                    weights=config.weights,
                )

                prev_weights = self.pm.current_weights.copy()
                weights_opt = self.pm.optimize(combined)

                total_tcost = self._calculate_tcosts(
                    weights_opt, prev_weights, config
                )
                gross_ret = self._simulate_interval_returns(
                    iids, weights_opt, ts, config, close_col
                )
                net_ret = gross_ret - total_tcost

                self.interval_results.append(
                    {
                        "timestamp": ts,
                        "gross_ret": gross_ret,
                        "net_ret": net_ret,
                        "tcost": total_tcost,
                    }
                )
                self.equity_curve.append(self.equity_curve[-1] * (1 + net_ret))

        return self.report(config.report_freq)

    def report(self, freq: str = "D") -> BacktestReport:
        if not self.interval_results:
            return {
                "status": self.status,
                "message": "No results generated.",
                "total_return": 0.0,
                "sharpe": 0.0,
                "drawdown": {"max_dd": 0.0},
                "final_equity": self.capital,
                "performance_table": pd.DataFrame(),
            }

        results_df = pd.DataFrame(self.interval_results)
        perf_table = PerformanceAnalyzer.generate_performance_table(
            results_df, freq=freq
        )

        # Full curve stats
        equity_series = pd.Series(self.equity_curve)
        net_returns = results_df["net_ret"]

        return {
            "status": self.status,
            "message": None,
            "total_return": self.equity_curve[-1] / self.equity_curve[0] - 1,
            "sharpe": PerformanceAnalyzer.calculate_sharpe(net_returns),
            "drawdown": PerformanceAnalyzer.calculate_drawdown(equity_series),
            "final_equity": self.equity_curve[-1],
            "performance_table": perf_table,
        }
