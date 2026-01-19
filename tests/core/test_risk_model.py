import numpy as np
import pytest

from src.core.portfolio_manager import PortfolioManager
from src.core.risk_model import RiskModel

# Constants to avoid magic values
STD_THRESHOLD = 0.5
EXPECTED_FACTOR_RET = 0.05
ASSET_1 = 1001
ASSET_2 = 1002


def test_risk_model_extracts_significant_factor_returns_via_pca() -> None:
    n_assets, n_obs = 5, 100
    returns = np.full((n_obs, 1), 0.01) @ np.ones(
        (n_assets, 1)
    ).T + np.random.normal(0, 0.001, (n_obs, n_assets))
    assert (
        np.std(RiskModel.get_factor_returns(returns, n_factors=1))
        > STD_THRESHOLD
    )


def test_risk_model_reconstructs_total_expected_returns_from_factors() -> None:
    pm, forecasts = PortfolioManager(), {ASSET_1: 0.0, ASSET_2: 0.0}
    returns_history = np.array([[0.01, 0.01], [0.02, 0.02], [-0.01, -0.01]])
    pm.update_risk_model(returns_history)
    pm.loadings = np.array([[1.0], [-1.0]])
    assert pytest.approx(sum(pm.optimize(forecasts).values())) == 0.0
    w_fact = pm.optimize(
        forecasts, factor_returns=np.array([EXPECTED_FACTOR_RET])
    )
    assert w_fact[ASSET_1] > 0 and w_fact[ASSET_2] < 0
