import unittest

import numpy as np

from src.core.portfolio_manager import PortfolioManager


class TestPortfolioExecutionFull(unittest.TestCase):
    def test_optimization_constraints(self) -> None:
        pm = PortfolioManager(risk_aversion=1.0)
        # Strong signal for asset 1, weak for asset 2
        forecasts = {1000: 0.1, 1001: 0.01}
        # 30 days of random returns for risk model
        returns = np.random.randn(30, 2) * 0.01

        weights = pm.optimize(forecasts, returns)

        # 1. Dollar Neutrality (Soft constraint, allow small residual)
        self.assertAlmostEqual(sum(weights.values()), 0.0, places=2)
        # 2. Leverage Limit (Soft constraint, allow small breach)
        self.assertLessEqual(sum(abs(w) for w in weights.values()), 1.2)
        # 3. Position Limit (Soft constraint)
        for w in weights.values():
            self.assertLessEqual(abs(w), 0.25)

    def test_safety_rate_limit(self) -> None:
        pm = PortfolioManager()
        pm.set_safety_limits(max_msgs=2, max_drawdown=-0.1)

        self.assertTrue(pm.check_safety(1.0))
        self.assertTrue(pm.check_safety(1.0))
        # Third within same second should fail
        self.assertFalse(pm.check_safety(1.0))

    def test_safety_kill_switch(self) -> None:
        pm = PortfolioManager()
        pm.set_safety_limits(max_msgs=10, max_drawdown=-0.05)  # 5% limit

        self.assertTrue(pm.check_safety(1.0))
        self.assertTrue(pm.check_safety(0.96))  # 4% drop - OK
        self.assertFalse(pm.check_safety(0.94))  # 6% drop - KILLED
        self.assertTrue(pm.killed)

        # Subsequent checks fail
        self.assertFalse(pm.check_safety(1.0))


if __name__ == "__main__":
    unittest.main()
