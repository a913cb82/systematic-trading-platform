from datetime import datetime
from typing import Dict, cast

import cvxpy as cp
import numpy as np


class RiskModel:
    @staticmethod
    def estimate_pca_covariance(
        returns: np.ndarray, n_factors: int = 5
    ) -> np.ndarray:
        """
        Decomposes returns into factor and specific risk using PCA.
        Sigma = B * Sigma_f * B^T + D
        """
        # 1. Standardize returns
        mu = returns.mean(axis=0)
        std = returns.std(axis=0)
        # Avoid division by zero
        std[std == 0] = 1.0
        z = (returns - mu) / std

        # 2. PCA via SVD
        # T (obs) x N (assets)
        _, s, vh = np.linalg.svd(z, full_matrices=False)

        # 3. Components (Factors)
        # s are singular values, eigenvalues = s^2 / (T-1)
        eigenvals = (s**2) / (returns.shape[0] - 1)
        # B is N x K (loadings)
        b = vh[:n_factors].T
        sigma_f = np.diag(eigenvals[:n_factors])

        # 4. Factor Covariance
        factor_cov = b @ sigma_f @ b.T

        # 5. Specific Risk (Diagonal)
        total_var = np.diag(np.cov(z, rowvar=False))
        # Ensure total_var is same shape as diagonal of factor_cov
        spec_var = np.maximum(total_var - np.diag(factor_cov), 0)
        d = np.diag(spec_var)

        # 6. Reconstruct Sigma in standardized space, then scale back
        sigma_z = factor_cov + d
        scale = np.outer(std, std)
        return cast(np.ndarray, sigma_z * scale)


class PortfolioManager:
    def __init__(  # noqa: PLR0913
        self,
        risk_aversion: float = 1.0,
        max_pos: float = 0.2,
        max_msgs_per_sec: int = 10,
        max_drawdown: float = -0.1,
        tc_penalty: float = 0.001,
        leverage_limit: float = 1.0,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.max_pos = max_pos
        self.tc_penalty = tc_penalty
        self.leverage_limit = leverage_limit
        self.current_weights: Dict[int, float] = {}

        # Soft Constraint Scalars (Lagrange Multipliers)
        self.lambda_net = 100.0  # Net exposure (neutrality)
        self.lambda_gross = 50.0  # Gross leverage
        self.lambda_pos = 10.0  # Individual position limits

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
        self,
        forecasts: Dict[int, float],
        returns_history: np.ndarray,
        use_pca: bool = False,
    ) -> Dict[int, float]:
        """
        Solves QP: Maximize Utility with Soft Penalties
        """
        try:
            iids = sorted(forecasts.keys())
            n = len(iids)
            if n == 0:
                return self.current_weights
            mu = np.array([forecasts[i] for i in iids])

            # 1. Covariance Estimation
            if use_pca:
                sigma = RiskModel.estimate_pca_covariance(returns_history)
            else:
                sigma = np.cov(returns_history, rowvar=False)
                # Ensure sigma is 2D
                if sigma.ndim == 0:
                    sigma = sigma.reshape((1, 1))
                elif sigma.ndim == 1:
                    sigma = np.diag(sigma)
                # Ledoit-Wolf style shrinkage
                sigma = 0.9 * sigma + 0.1 * np.eye(n) * np.trace(sigma) / n
                sigma += 1e-6 * np.eye(n)

            w = cp.Variable(n)
            prev_w = np.array([self.current_weights.get(i, 0.0) for i in iids])

            # 2. Objective Components
            risk = cp.quad_form(w, cp.psd_wrap(sigma))
            # Linear Transaction Costs
            tc = self.tc_penalty * cp.norm(w - prev_w, 1)
            # Market Impact (Square Root Law approx: Power 1.5 for cost)
            impact = 0.005 * cp.sum(cp.power(cp.abs(w - prev_w), 1.5))

            # 3. Soft Constraints (Lagrange Multipliers)
            # Net Exposure Penalty (Neutrality)
            net_penalty = self.lambda_net * cp.square(cp.sum(w))
            # Gross Exposure Penalty (Leverage)
            gross_penalty = self.lambda_gross * cp.square(
                cp.pos(cp.norm(w, 1) - self.leverage_limit)
            )
            # Individual Position Penalties (instead of hard limits)
            pos_penalty = self.lambda_pos * cp.sum(
                cp.square(cp.pos(cp.abs(w) - self.max_pos))
            )

            obj = cp.Maximize(
                w @ mu
                - 0.5 * self.risk_aversion * risk
                - tc
                - impact
                - net_penalty
                - gross_penalty
                - pos_penalty
            )

            # No hard constraints - improves solver robustness
            prob = cp.Problem(obj)
            prob.solve()

            if w.value is None:
                return self.current_weights

            self.current_weights = {
                iids[i]: float(w.value[i]) for i in range(n)
            }
        except Exception:
            return self.current_weights

        return self.current_weights
