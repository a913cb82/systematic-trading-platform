from datetime import datetime
from typing import Dict

import cvxpy as cp
import numpy as np

from ..common.base import PortfolioOptimizer, RiskModel
from ..common.config import config


class SimpleOptimizer(PortfolioOptimizer):
    def optimize(
        self,
        timestamp: datetime,
        forecasts: Dict[int, float],
        current_weights: Dict[int, float] | None = None,
    ) -> Dict[int, float]:
        """
        Simple optimizer that treats forecasts as target weights.
        Normalizes signals to sum to 1.0.
        """
        if not forecasts:
            return {}

        total_signal = sum(forecasts.values())
        if total_signal == 0:
            # If all signals are 0 or sum to 0, equal weight
            return {iid: 1.0 / len(forecasts) for iid in forecasts}

        return {iid: s / total_signal for iid, s in forecasts.items()}


class CvxpyOptimizer(PortfolioOptimizer):
    def __init__(  # noqa: PLR0913
        self,
        risk_model: RiskModel,
        lambda_risk: float | None = None,
        max_position: float | None = None,
        max_turnover: float | None = None,
        market_impact_coeff: float = 0.1,
        lambda_sector: float = 0.0,
        lambda_market: float = 0.0,
    ):
        self.risk_model = risk_model
        self.lambda_risk = (
            lambda_risk
            if lambda_risk is not None
            else config.get("portfolio.risk_aversion", 1.0)
        )
        self.max_position = (
            max_position
            if max_position is not None
            else config.get("portfolio.max_position", 1.0)
        )
        self.max_turnover = (
            max_turnover
            if max_turnover is not None
            else config.get("portfolio.max_turnover", 1.0)
        )
        self.market_impact_coeff = market_impact_coeff
        self.lambda_sector = lambda_sector
        self.lambda_market = lambda_market

    def optimize(
        self,
        timestamp: datetime,
        forecasts: Dict[int, float],
        current_weights: Dict[int, float] | None = None,
    ) -> Dict[int, float]:
        if not forecasts:
            return {}

        internal_ids = sorted(forecasts.keys())
        n = len(internal_ids)
        mu = np.array([forecasts[iid] for iid in internal_ids])

        # Current weights vector
        w_prev = np.zeros(n)
        if current_weights:
            for i, iid in enumerate(internal_ids):
                w_prev[i] = current_weights.get(iid, 0.0)

        # Get covariance matrix
        cov_matrix = np.array(
            self.risk_model.get_covariance_matrix(timestamp, internal_ids)
        )

        # Define CVXPY problem
        w = cp.Variable(n)

        # Objective: Maximize w^T * mu - 0.5 * lambda * w^T * Sigma * w
        risk = cp.quad_form(w, cov_matrix)

        # Linear Transaction costs (turnover penalty)
        turnover = cp.norm(w - w_prev, 1)

        # Non-linear Market Impact (Square Root Law proxy: power 1.5)
        impact = cp.sum(cp.power(cp.abs(w - w_prev), 1.5))

        # Soft constraints penalties
        soft_penalties = 0

        # Market Neutrality (Net Exposure target 0)
        if self.lambda_market > 0:
            soft_penalties += self.lambda_market * cp.square(cp.sum(w))

        # Sector Neutrality
        if self.lambda_sector > 0:
            exposures = self.risk_model.get_factor_exposures(
                timestamp, internal_ids
            )
            sectors = sorted(
                list(
                    set(
                        exp.get("sector", "Unknown")
                        for exp in exposures.values()
                    )
                )
            )
            for sector in sectors:
                # Exposure to this sector
                exp_vec = np.array(
                    [
                        1.0 if exposures[iid].get("sector") == sector else 0.0
                        for iid in internal_ids
                    ]
                )
                soft_penalties += self.lambda_sector * cp.square(w @ exp_vec)

        # Objective function
        objective = cp.Maximize(
            w @ mu
            - 0.5 * self.lambda_risk * risk
            - 0.01 * turnover
            - self.market_impact_coeff * impact
            - soft_penalties
        )

        # Constraints
        # Note: If market neutral is requested, sum(w) == 1 might be wrong.
        # Usually market neutral means sum(w) == 0.
        # Let's adjust based on lambda_market.
        max_leverage = 2.0
        if self.lambda_market > 0:
            constraints = [
                w >= -self.max_position,
                w <= self.max_position,
                cp.norm(w, 1) <= max_leverage,  # Limit gross leverage
            ]
        else:
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= self.max_position,
            ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except Exception:
            # Fallback
            return SimpleOptimizer().optimize(timestamp, forecasts)

        if (
            prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
            or w.value is None
        ):
            return SimpleOptimizer().optimize(timestamp, forecasts)

        return {internal_ids[i]: float(w.value[i]) for i in range(n)}
