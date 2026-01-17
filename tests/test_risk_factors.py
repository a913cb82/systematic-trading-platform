import unittest

import numpy as np

from src.core.portfolio_manager import PortfolioManager
from src.core.risk_model import RiskModel


class TestRiskFactors(unittest.TestCase):
    def test_factor_return_calculation(self) -> None:
        # Create returns where the first factor (market) is clearly 1%
        # returns = factors @ loadings.T + noise
        n_assets = 5
        n_obs = 100
        loadings = np.ones(
            (n_assets, 1)
        )  # All assets have beta 1.0 to factor 1
        factor_rets = np.full((n_obs, 1), 0.01)
        # Add noise to give PCA something to work with
        noise = np.random.normal(0, 0.001, (n_obs, n_assets))
        returns = factor_rets @ loadings.T + noise

        calculated_f_rets = RiskModel.get_factor_returns(returns, n_factors=1)
        # Factor 1 should capture the 1% return.
        # Mean of z-scored data is 0, but variance should be high.
        min_std = 0.5
        self.assertTrue(np.std(calculated_f_rets) > min_std)

    def test_total_return_reconstruction(self) -> None:
        pm = PortfolioManager()
        # 2 assets, 1 factor
        forecasts = {1001: 0.0, 1002: 0.0}  # Zero idiosyncratic alpha
        returns_history = np.array(
            [[0.01, 0.01], [0.02, 0.02], [-0.01, -0.01]]
        )

        # Manually set loadings: asset 1 has beta 1.0, asset 2 has beta -1.0
        pm.update_risk_model(returns_history)
        pm.loadings = np.array([[1.0], [-1.0]])

        # Factor expected return is 5%
        factor_returns = np.array([0.05])

        # Without factor returns, weights should be 0 (since mu=0)
        w_zero = pm.optimize(forecasts)
        self.assertAlmostEqual(sum(w_zero.values()), 0.0)

        # With factor returns, Asset 1 (beta 1) long, Asset 2 (beta -1) short
        w_fact = pm.optimize(forecasts, factor_returns=factor_returns)
        self.assertTrue(w_fact[1001] > 0)
        self.assertTrue(w_fact[1002] < 0)


if __name__ == "__main__":
    unittest.main()
