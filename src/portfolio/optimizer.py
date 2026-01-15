from datetime import datetime
from typing import Dict, List
import numpy as np
import cvxpy as cp
from ..common.base import PortfolioOptimizer, RiskModel

class SimpleOptimizer(PortfolioOptimizer):
    def optimize(self, timestamp: datetime, forecasts: Dict[int, float], current_weights: Dict[int, float] | None = None) -> Dict[int, float]:
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
    def __init__(self, risk_model: RiskModel, lambda_risk: float = 1.0, 
                 max_position: float = 1.0, max_turnover: float = 1.0):
        self.risk_model = risk_model
        self.lambda_risk = lambda_risk
        self.max_position = max_position
        self.max_turnover = max_turnover

    def optimize(self, timestamp: datetime, forecasts: Dict[int, float], 
                 current_weights: Dict[int, float] | None = None) -> Dict[int, float]:
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
        cov_matrix = np.array(self.risk_model.get_covariance_matrix(timestamp, internal_ids))
        
        # Define CVXPY problem
        w = cp.Variable(n)
        
        # Objective: Maximize w^T * mu - 0.5 * lambda * w^T * Sigma * w
        risk = cp.quad_form(w, cov_matrix)
        
        # Transaction costs (simplified as linear turnover penalty)
        turnover = cp.norm(w - w_prev, 1)
        
        # Objective function
        objective = cp.Maximize(w @ mu - 0.5 * self.lambda_risk * risk - 0.01 * turnover)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_position
        ]
        
        # Hard turnover constraint if requested
        if self.max_turnover < 1.0:
            constraints.append(turnover <= self.max_turnover)
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except Exception:
            # Fallback
            return SimpleOptimizer().optimize(timestamp, forecasts)
        
        if prob.status != cp.OPTIMAL and prob.status != cp.OPTIMAL_INACCURATE:
            return SimpleOptimizer().optimize(timestamp, forecasts)
            
        return {internal_ids[i]: float(w.value[i]) for i in range(n)}
