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
        factor_returns: Dict[str, float] | None = None,
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
        lambda_leverage: float = 10.0,
        target_leverage: float = 1.0,
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
        self.lambda_leverage = lambda_leverage
        self.target_leverage = target_leverage

    def _reconstruct_mu(
        self,
        timestamp: datetime,
        internal_ids: list[int],
        forecasts: Dict[int, float],
        factor_returns: Dict[str, float] | None,
    ) -> np.ndarray:
        mu = np.array([forecasts[iid] for iid in internal_ids])
        if factor_returns:
            exposures = self.risk_model.get_factor_exposures(
                timestamp, internal_ids
            )
            for i, iid in enumerate(internal_ids):
                asset_exp = exposures.get(iid, {})
                for factor, beta in asset_exp.items():
                    if factor in factor_returns:
                        mu[i] += beta * factor_returns[factor]
        return mu

    def _get_soft_penalties(
        self,
        timestamp: datetime,
        internal_ids: list[int],
        w: cp.Variable,
    ) -> float:
        soft_penalties = 0
        if self.lambda_market > 0:
            soft_penalties += self.lambda_market * cp.square(cp.sum(w))

        if self.lambda_sector > 0:
            exposures = self.risk_model.get_factor_exposures(
                timestamp, internal_ids
            )
            sectors: set[str] = set()
            for exp in exposures.values():
                sectors.update(exp.keys())
            for sector in sorted(list(sectors)):
                exp_vec = np.array(
                    [exposures[iid].get(sector, 0.0) for iid in internal_ids]
                )
                soft_penalties += self.lambda_sector * cp.square(w @ exp_vec)

        if self.lambda_leverage > 0:
            soft_penalties += self.lambda_leverage * cp.square(
                cp.norm(w, 1) - self.target_leverage
            )
        return soft_penalties

    def optimize(
        self,
        timestamp: datetime,
        forecasts: Dict[int, float],
        current_weights: Dict[int, float] | None = None,
        factor_returns: Dict[str, float] | None = None,
    ) -> Dict[int, float]:
        if not forecasts:
            return {}

        internal_ids = sorted(forecasts.keys())
        n = len(internal_ids)
        mu = self._reconstruct_mu(
            timestamp, internal_ids, forecasts, factor_returns
        )

        w_prev = np.zeros(n)
        if current_weights:
            for i, iid in enumerate(internal_ids):
                w_prev[i] = current_weights.get(iid, 0.0)

        cov_matrix = np.array(
            self.risk_model.get_covariance_matrix(timestamp, internal_ids)
        )

        w = cp.Variable(n)
        risk = cp.quad_form(w, cov_matrix)
        turnover = cp.norm(w - w_prev, 1)
        impact = cp.sum(cp.power(cp.abs(w - w_prev), 1.5))
        soft_penalties = self._get_soft_penalties(timestamp, internal_ids, w)

        objective = cp.Maximize(
            w @ mu
            - 0.5 * self.lambda_risk * risk
            - 0.01 * turnover
            - self.market_impact_coeff * impact
            - soft_penalties
        )

        constraints = [w >= -self.max_position, w <= self.max_position]
        if self.max_turnover < 1.0:
            constraints.append(cp.norm(w - w_prev, 1) <= self.max_turnover)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except Exception:
            return SimpleOptimizer().optimize(timestamp, forecasts)

        if (
            prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
            or w.value is None
        ):
            return SimpleOptimizer().optimize(timestamp, forecasts)

        return {internal_ids[i]: float(w.value[i]) for i in range(n)}
