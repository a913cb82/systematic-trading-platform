from typing import Dict

import numpy as np
import numpy.typing as npt
import pandas as pd


class PerformanceAnalyzer:
    @staticmethod
    def calculate_sharpe(
        returns: pd.Series, freq_multiplier: float = 252
    ) -> float:
        if returns.empty or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(freq_multiplier))

    @staticmethod
    def calculate_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        if equity_curve.empty:
            return {"max_dd": 0.0}
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return {
            "max_dd": float(drawdown.min()),
            "current_dd": float(drawdown.iloc[-1]),
        }

    @staticmethod
    def factor_attribution(
        portfolio_weights: Dict[int, float],
        asset_returns: Dict[int, float],
        factor_loadings: npt.NDArray[np.float64],  # N x K
    ) -> Dict[str, float]:
        """
        Decomposes a single-period return using PCA components.
        """
        iids = sorted(portfolio_weights.keys())
        w = np.array([portfolio_weights[i] for i in iids])
        r = np.array([asset_returns.get(i, 0.0) for i in iids])

        total_return = w @ r

        # Factor Returns (project realized returns onto loadings)
        # since B is orthogonal from PCA, f = B' r
        f = factor_loadings.T @ r

        # Factor Contribution = w' * B * f
        contribution_from_factors = w @ factor_loadings @ f
        selection_alpha = total_return - contribution_from_factors

        return {
            "total": float(total_return),
            "factor": float(contribution_from_factors),
            "selection": float(selection_alpha),
        }

    @staticmethod
    def generate_performance_table(
        returns_df: pd.DataFrame, freq: str = "D"
    ) -> pd.DataFrame:
        """
        Aggregates returns and calculates metrics per period.
        Expected columns in returns_df: ['timestamp', 'gross_ret', 'net_ret']
        freq can be 'D', 'W', 'M', 'Y'
        """
        df = returns_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Define aggregation logic
        resampled = df.resample(freq)

        def calc_metrics(group: pd.DataFrame) -> pd.Series:
            if group.empty:
                return pd.Series(dtype=float)

            # Cumulative returns for the period
            gross_cum = (1 + group["gross_ret"]).prod() - 1
            net_cum = (1 + group["net_ret"]).prod() - 1

            # Multiplier for annualization
            ann_factor = (
                252
                if freq == "D"
                else (52 if freq == "W" else (12 if freq == "M" else 1))
            )

            g_sharpe = PerformanceAnalyzer.calculate_sharpe(
                group["gross_ret"], freq_multiplier=ann_factor
            )
            n_sharpe = PerformanceAnalyzer.calculate_sharpe(
                group["net_ret"], freq_multiplier=ann_factor
            )

            # Max Drawdown for the period
            equity = (1 + group["net_ret"]).cumprod()
            mdd = PerformanceAnalyzer.calculate_drawdown(equity)["max_dd"]

            return pd.Series(
                {
                    "Return (Net)": f"{net_cum:.2%}",
                    "Return (Gross)": f"{gross_cum:.2%}",
                    "Ann. Sharpe (Net)": f"{n_sharpe:.2f}",
                    "Ann. Sharpe (Gross)": f"{g_sharpe:.2f}",
                    "Max Drawdown": f"{mdd:.2%}",
                }
            )

        resampled_metrics = resampled.apply(calc_metrics)

        # Calculate summary row
        summary_metrics = calc_metrics(df)
        summary_metrics.name = "Summary"

        return pd.concat([resampled_metrics, summary_metrics.to_frame().T])
