import unittest
from unittest.mock import MagicMock

import numpy as np

from src.core.execution_handler import ExecutionHandler
from src.core.portfolio_manager import PortfolioManager


class TestPortfolioExecution(unittest.TestCase):
    def test_rebalance_integration(self) -> None:
        """Verify pm.optimize outputs can be passed to execution.rebalance."""
        backend = MagicMock()
        backend.get_positions.return_value = {"AAPL": 0.0}
        backend.get_prices.return_value = {"AAPL": 150.0}

        pm = PortfolioManager()
        forecasts = {1000: 0.05}
        returns = np.random.randn(20, 1) * 0.01
        weights = pm.optimize(forecasts, returns)

        handler = ExecutionHandler(backend)
        # Goal position calculation (weight * capital / price)
        goal_positions = {"AAPL": (weights[1000] * 100000) / 150.0}

        # This just verifies no type errors or crashes
        handler.rebalance(goal_positions, interval=0)


if __name__ == "__main__":
    unittest.main()
