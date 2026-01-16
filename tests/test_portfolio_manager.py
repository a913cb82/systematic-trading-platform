import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from src.core.portfolio_manager import PortfolioManager


class TestPortfolioManagerAdditional(unittest.TestCase):
    def test_rate_limit_reset(self) -> None:
        pm = PortfolioManager()
        pm.set_safety_limits(max_msgs=1, max_drawdown=-0.1)
        self.assertTrue(pm.check_safety(1.0))
        self.assertFalse(pm.check_safety(1.0))

        future_time = datetime.now() + timedelta(seconds=2)
        with patch("src.core.portfolio_manager.datetime") as mock_dt:
            mock_dt.now.return_value = future_time
            self.assertTrue(pm.check_safety(1.0))
            self.assertEqual(pm.msg_count, 1)

    def test_singular_cov_robustness(self) -> None:
        pm = PortfolioManager()
        forecasts = {1000: 0.1}
        # Scalar covariance
        returns = np.array([[0.01], [0.011]])
        weights = pm.optimize(forecasts, returns)
        self.assertIn(1000, weights)

    def test_tc_impact_on_optimization(self) -> None:
        # High TC penalty should prevent small rebalances
        pm = PortfolioManager(risk_aversion=0.1, tc_penalty=0.1)
        forecasts = {1000: 0.1}
        returns = np.array([[0.01], [0.011]])

        w1 = pm.optimize(forecasts, returns)[1000]

        forecasts_new = {1000: 0.11}  # Small change
        w2 = pm.optimize(forecasts_new, returns)[1000]

        self.assertAlmostEqual(w1, w2, places=4)


if __name__ == "__main__":
    unittest.main()
