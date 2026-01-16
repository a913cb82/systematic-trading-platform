from datetime import datetime
from typing import Dict, Optional

import cvxpy as cp
import numpy as np

from src.core.risk_model import RiskModel


class PortfolioManager:
    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_pos: float = 0.2,
        tc_penalty: float = 0.001,
        leverage_limit: float = 1.0,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.max_pos = max_pos
        self.tc_penalty = tc_penalty
        self.leverage_limit = leverage_limit
        self.current_weights: Dict[int, float] = {}
        self.sigma: Optional[np.ndarray] = None
        self.loadings: Optional[np.ndarray] = None

        # Soft Constraint Scalars (Lagrange Multipliers)
        self.lambda_net = 100.0  # Net exposure (neutrality)
        self.lambda_gross = 50.0  # Gross leverage
        self.lambda_pos = 10.0  # Individual position limits

        # Safety Defaults
        self.msg_count = 0
        self.last_msg_ts = datetime.now()
        self.max_msgs = 10
        self.max_dd = -0.1
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.killed = False

    def set_safety_limits(self, max_msgs: int, max_drawdown: float) -> None:
        self.max_msgs = max_msgs
        self.max_dd = max_drawdown

    def update_risk_model(self, returns_history: np.ndarray) -> None:
        """
        Calculates and caches the PCA-based risk parameters.
        Should be called once at the start of the trading day.
        """
        self.sigma, self.loadings = RiskModel.estimate_pca_covariance(
            returns_history
        )

    def check_safety(self, equity: float) -> bool:
        if self.killed:
            return False
        # Kill Switch
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if (equity - self.peak_equity) / self.peak_equity < self.max_dd:
            self.killed = True
            return False
        # Rate Limit
        now = datetime.now()
        if (now - self.last_msg_ts).total_seconds() < 1.0:
            self.msg_count += 1
        else:
            self.msg_count = 1
            self.last_msg_ts = now
        return self.msg_count <= self.max_msgs

    def optimize(
        self,
        forecasts: Dict[int, float],
        returns_history: Optional[np.ndarray] = None,
    ) -> Dict[int, float]:
        """
        Solves QP: Maximize Utility using cached PCA Risk Model.
        """
        try:
            iids = sorted(forecasts.keys())
            n = len(iids)
            if n == 0:
                return self.current_weights

            # 1. Covariance Retrieval & Fallback
            if self.sigma is None or self.sigma.shape[0] != n:
                if returns_history is None:
                    return self.current_weights
                self.update_risk_model(returns_history)

            sigma = self.sigma
            if sigma is None:
                return self.current_weights

            mu = np.array([forecasts[i] for i in iids])
            w = cp.Variable(n)
            prev_w = np.array([self.current_weights.get(i, 0.0) for i in iids])

            # 2. Objective Components
            risk = cp.quad_form(w, cp.psd_wrap(sigma))
            tc = self.tc_penalty * cp.norm(w - prev_w, 1)
            impact = 0.005 * cp.sum(cp.power(cp.abs(w - prev_w), 1.5))

            # 3. Soft Constraints
            net_pen = self.lambda_net * cp.square(cp.sum(w))
            gross_pen = self.lambda_gross * cp.square(
                cp.pos(cp.norm(w, 1) - self.leverage_limit)
            )
            pos_pen = self.lambda_pos * cp.sum(
                cp.square(cp.pos(cp.abs(w) - self.max_pos))
            )

            obj = cp.Maximize(
                w @ mu
                - 0.5 * self.risk_aversion * risk
                - tc
                - impact
                - net_pen
                - gross_pen
                - pos_pen
            )

            cp.Problem(obj).solve()

            if w.value is not None:
                self.current_weights = {
                    iids[i]: float(w.value[i]) for i in range(n)
                }
        except Exception:
            pass

        return self.current_weights
