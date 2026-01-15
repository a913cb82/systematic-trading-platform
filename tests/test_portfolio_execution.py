import unittest

import numpy as np

from src.portfolio_manager import PortfolioManager


class TestPortfolioExecutionFull(unittest.TestCase):
    def test_optimization_constraints(self) -> None:
        pm = PortfolioManager(risk_aversion=1.0)
        # Strong signal for asset 1, weak for asset 2
        forecasts = {1000: 0.1, 1001: 0.01}
        # 30 days of random returns for risk model
        returns = np.random.randn(30, 2) * 0.01

        weights = pm.optimize(forecasts, returns)

        # 1. Dollar Neutrality
        self.assertAlmostEqual(sum(weights.values()), 0.0, places=5)
        # 2. Leverage Limit (Sum of abs weights <= 1.0)
        self.assertLessEqual(sum(abs(w) for w in weights.values()), 1.0001)
        # 3. Position Limit (0.2)
        for w in weights.values():
            self.assertLessEqual(abs(w), 0.2001)

    def test_safety_rate_limit(self) -> None:
        pm = PortfolioManager(max_msgs_per_sec=2)

        self.assertTrue(pm.check_safety(1.0))
        self.assertTrue(pm.check_safety(1.0))
        # Third within same second should fail
        self.assertFalse(pm.check_safety(1.0))

    def test_safety_kill_switch(self) -> None:
        pm = PortfolioManager(max_drawdown=-0.05)  # 5% limit

        self.assertTrue(pm.check_safety(1.0))
        self.assertTrue(pm.check_safety(0.96))  # 4% drop - OK
        self.assertFalse(pm.check_safety(0.94))  # 6% drop - KILLED
        self.assertTrue(pm.killed)

        # Subsequent checks fail
        self.assertFalse(pm.check_safety(1.0))


if __name__ == "__main__":
    unittest.main()
