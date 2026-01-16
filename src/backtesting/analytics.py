from typing import Dict

import numpy as np
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
        factor_loadings: np.ndarray,  # N x K
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
