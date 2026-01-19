from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from src.core.portfolio_manager import PortfolioManager

# Constants to avoid magic values
MAX_MSGS = 1
DRAWDOWN_LIMIT = -0.1
TEST_IID = 1000
TEST_IID_2 = 1001
LEVERAGE_LIMIT = 0.5
KILL_SWITCH_DRAWDOWN = -0.05
TRIGGER_DRAWDOWN = 0.94
LEVERAGE_UPPER_BOUND = 2.0


@pytest.fixture
def pm() -> PortfolioManager:
    return PortfolioManager()


def test_portfoliomanager_enforces_safety_limits_on_messages(
    pm: PortfolioManager,
) -> None:
    pm.set_safety_limits(max_msgs=MAX_MSGS, max_drawdown=DRAWDOWN_LIMIT)
    assert pm.check_safety(1.0)
    assert not pm.check_safety(1.0)
    future_time = datetime.now() + timedelta(seconds=2)
    with patch("src.core.portfolio_manager.datetime") as mock_dt:
        mock_dt.now.return_value = future_time
        assert pm.check_safety(1.0)


def test_portfoliomanager_handles_singular_covariance_matrix(
    pm: PortfolioManager,
) -> None:
    forecasts, returns = {TEST_IID: 0.1}, np.array([[0.01], [0.011]])
    assert TEST_IID in pm.optimize(forecasts, returns)


def test_portfoliomanager_penalizes_trades_via_transaction_costs(
    pm: PortfolioManager,
) -> None:
    pm.risk_aversion, pm.tc_penalty = 0.1, 0.1
    returns = np.array([[0.01], [0.011]])
    w1 = pm.optimize({TEST_IID: 0.1}, returns)[TEST_IID]
    w2 = pm.optimize({TEST_IID: 0.11}, returns)[TEST_IID]
    assert pytest.approx(w1, abs=1e-4) == w2


def test_portfoliomanager_returns_reasonable_solution_for_infeasible_goals(
    pm: PortfolioManager,
) -> None:
    pm.leverage_limit = LEVERAGE_LIMIT
    forecasts, returns = (
        {TEST_IID: 1.0, TEST_IID_2: 1.0},
        np.random.randn(20, 2) * 0.01,
    )
    weights = pm.optimize(forecasts, returns)
    assert sum(abs(v) for v in weights.values()) < LEVERAGE_UPPER_BOUND


def test_portfoliomanager_handles_missing_stale_risk_data(
    pm: PortfolioManager,
) -> None:
    assert pm.optimize({TEST_IID: 0.1}, None) == {}


def test_portfoliomanager_optimizes_with_sufficient_history(
    pm: PortfolioManager,
) -> None:
    forecasts, returns = {TEST_IID: 0.1}, np.random.randn(20, 1) * 0.01
    assert TEST_IID in pm.optimize(forecasts, returns)


def test_portfoliomanager_gracefully_handles_invalid_return_history(
    pm: PortfolioManager,
) -> None:
    invalid_returns = np.array([[np.nan, 0.1], [0.1, np.nan]])
    weights = pm.optimize({TEST_IID: 0.1, TEST_IID_2: 0.2}, invalid_returns)
    assert isinstance(weights, dict)


def test_portfoliomanager_activates_kill_switch_on_excessive_drawdown(
    pm: PortfolioManager,
) -> None:
    pm.set_safety_limits(max_msgs=10, max_drawdown=KILL_SWITCH_DRAWDOWN)
    assert pm.check_safety(1.0)
    assert not pm.check_safety(TRIGGER_DRAWDOWN)
    assert pm.killed


def test_portfoliomanager_ignores_optimization_without_forecasts(
    pm: PortfolioManager,
) -> None:
    assert pm.optimize({}) == {}


def test_portfoliomanager_fails_without_sufficient_risk_parameters(
    pm: PortfolioManager,
) -> None:
    pm.sigma = None
    assert pm.optimize({TEST_IID: 0.1}, returns_history=None) == {}
