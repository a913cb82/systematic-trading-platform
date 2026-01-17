import unittest

import numpy as np

from src.core.portfolio_manager import PortfolioManager


class TestPortfolioRobustness(unittest.TestCase):
    def setUp(self) -> None:
        self.pm = PortfolioManager()

    def test_infeasible_soft_constraints(self) -> None:
        """
        Verify that even with impossible goals (leverage > limit),
        the soft constraint solver returns a reasonable solution.
        """
        self.pm.leverage_limit = 0.5
        # Large forecasts that want to push leverage high
        forecasts = {1000: 1.0, 1001: 1.0}
        returns = np.random.randn(20, 2) * 0.01

        weights = self.pm.optimize(forecasts, returns)
        # Should return a result, likely capped near leverage_limit
        self.assertTrue(len(weights) > 0)

        # Check leverage is within reasonable bounds
        total_lev = sum(abs(v) for v in weights.values())
        leverage_upper_bound = 2.0
        self.assertTrue(total_lev < leverage_upper_bound)

    def test_risk_model_fallback(self) -> None:
        """Verify optimizer handles missing or stale risk data gracefully."""
        # 1. No risk model yet
        forecasts = {1000: 0.1}
        # Positional arg check
        w1 = self.pm.optimize(forecasts, None)
        self.assertEqual(w1, {})

        # 2. Update with history
        returns = np.random.randn(20, 1) * 0.01
        w2 = self.pm.optimize(forecasts, returns)

        target_id = 1000
        self.assertTrue(target_id in w2)

    def test_invalid_return_history(self) -> None:
        """Test optimizer behavior with NaN or invalid returns."""
        invalid_returns = np.array([[np.nan, 0.1], [0.1, np.nan]])
        # Should not crash, returns previous or empty
        weights = self.pm.optimize({1000: 0.1, 1001: 0.2}, invalid_returns)
        self.assertIsInstance(weights, dict)

    def test_kill_switch(self) -> None:
        """Test the kill switch safety limit."""
        self.pm.set_safety_limits(max_msgs=10, max_drawdown=-0.05)
        # Trigger drawdown
        self.assertTrue(self.pm.check_safety(1.0))
        self.assertFalse(self.pm.check_safety(0.94))
        self.assertTrue(self.pm.killed)

        # Once killed, everything stays killed
        self.assertFalse(self.pm.check_safety(1.0))


if __name__ == "__main__":
    unittest.main()
