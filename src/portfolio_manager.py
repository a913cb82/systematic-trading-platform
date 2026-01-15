from datetime import datetime
from typing import Dict

import cvxpy as cp
import numpy as np


class PortfolioManager:
    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_pos: float = 0.2,
        max_msgs_per_sec: int = 10,
        max_drawdown: float = -0.1,
        tc_penalty: float = 0.001,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.max_pos = max_pos
        self.tc_penalty = tc_penalty
        self.current_weights: Dict[int, float] = {}
        # Safety
        self.msg_count = 0
        self.last_msg_ts = datetime.now()
        self.max_msgs = max_msgs_per_sec
        self.max_dd = max_drawdown
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.killed = False

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
        self, forecasts: Dict[int, float], returns_history: np.ndarray
    ) -> Dict[int, float]:
        """
        Solves QP: Maximize w'mu - 0.5 * lambda * w'Sigma*w - tc_penalty
        """
        try:
            iids = sorted(forecasts.keys())
            n = len(iids)
            if n == 0:
                return self.current_weights
            mu = np.array([forecasts[i] for i in iids])

            # 1. Shrinkage Covariance (Robustness requirement)
            sigma = np.cov(returns_history, rowvar=False)
            # Ensure sigma is 2D
            if sigma.ndim == 0:
                sigma = sigma.reshape((1, 1))
            elif sigma.ndim == 1:
                sigma = np.diag(sigma)

            sigma = 0.9 * sigma + 0.1 * np.eye(n) * np.trace(sigma) / n

            w = cp.Variable(n)
            prev_w = np.array([self.current_weights.get(i, 0.0) for i in iids])

            # 2. Objective with Transaction Costs (Linear)
            risk = cp.quad_form(w, sigma)
            tc = self.tc_penalty * cp.norm(w - prev_w, 1)

            prob = cp.Problem(
                cp.Maximize(w @ mu - 0.5 * self.risk_aversion * risk - tc),
                [
                    cp.sum(w) == 0,  # Dollar Neutral
                    cp.norm(w, 1) <= 1.0,  # Leverage Limit
                    w >= -self.max_pos,
                    w <= self.max_pos,
                ],
            )

            prob.solve()
            if w.value is None:
                return self.current_weights

            self.current_weights = {
                iids[i]: float(w.value[i]) for i in range(n)
            }
        except Exception:
            # Return current weights on any failure (infeasibility, etc.)
            return self.current_weights

        return self.current_weights
