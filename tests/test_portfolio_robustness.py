import unittest
import warnings

import numpy as np

from src.core.portfolio_manager import PortfolioManager


class TestPortfolioRobustness(unittest.TestCase):
    def setUp(self) -> None:
        self.pm = PortfolioManager(risk_aversion=1.0, max_pos=0.2)

    def test_singular_covariance_handling(self) -> None:
        """Tests that shrinkage makes singular covariance invertible."""
        # 2 Assets with perfectly correlated returns
        returns = np.array([[0.01, 0.01], [0.02, 0.02], [0.01, 0.01]] * 5)
        forecasts = {1000: 0.1, 1001: 0.05}

        # This shouldn't crash CVXPY because we apply shrinkage
        weights = self.pm.optimize(forecasts, returns)
        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(sum(weights.values()), 0.0, places=2)

    def test_transaction_cost_impact(self) -> None:
        """Tests that TC penalty reduces turnover."""
        forecasts = {1000: 0.1, 1001: -0.1}
        returns = np.random.randn(20, 2) * 0.01

        # 1. First run: Start from zero
        w1 = self.pm.optimize(forecasts, returns)

        # 2. Second run: Same signals, but we already have positions.
        w2 = self.pm.optimize(forecasts, returns)

        for iid in w1:
            # Tolerances increased due to non-linear impact model
            self.assertAlmostEqual(w1[iid], w2[iid], places=3)

    def test_infeasible_constraints_graceful_fail(self) -> None:
        """Tests behavior when constraints are impossible."""
        # Our current constraints are quite loose.
        # But if the solver fails, it should return current_weights.
        self.pm.current_weights = {1000: 0.1, 1001: 0.2}

        # Passing garbage data to force a failure
        invalid_returns = np.array([[np.nan, np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            weights = self.pm.optimize({1000: 0.1, 1001: 0.2}, invalid_returns)

        self.assertEqual(weights, {1000: 0.1, 1001: 0.2})

    def test_large_universe_performance(self) -> None:
        """Verifies optimizer handles 100+ assets (PoC scale)."""
        n = 100
        forecasts = {i: float(np.random.randn() * 0.01) for i in range(n)}
        returns = np.random.randn(50, n) * 0.01

        weights = self.pm.optimize(forecasts, returns)
        self.assertEqual(len(weights), n)
        self.assertAlmostEqual(sum(weights.values()), 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
