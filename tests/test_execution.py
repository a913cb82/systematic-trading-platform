import unittest
from unittest.mock import MagicMock

from src.execution_handler import ExecutionHandler


class TestExecutionHandler(unittest.TestCase):
    def test_rebalance_basic(self) -> None:
        backend = MagicMock()
        backend.get_positions.return_value = {"AAPL": 10.0}
        backend.get_prices.return_value = {"AAPL": 100.0}

        handler = ExecutionHandler(backend)
        # 50% of 10000 = 5000 / 100 = 50 shares. Diff = +40
        target_weights = {1001: 0.5}
        reverse_ism = {1001: "AAPL"}

        handler.rebalance(target_weights, reverse_ism, 10000.0)
        backend.submit_order.assert_called_with("AAPL", 40.0, "BUY")

    def test_rebalance_multi_asset(self) -> None:
        backend = MagicMock()
        backend.get_positions.return_value = {"AAPL": 100.0, "MSFT": 50.0}
        backend.get_prices.return_value = {"AAPL": 100.0, "MSFT": 200.0}

        handler = ExecutionHandler(backend)
        # Target: AAPL 100 shares (no change), MSFT 60 shares (+10)
        # Capital 22000.
        target_weights = {1001: 0.454545, 1002: 0.545454}
        reverse_ism = {1001: "AAPL", 1002: "MSFT"}

        handler.rebalance(target_weights, reverse_ism, 22000.0)

        # Should only call submit_order for MSFT
        backend.submit_order.assert_called_once()
        args, _ = backend.submit_order.call_args
        self.assertEqual(args[0], "MSFT")
        self.assertAlmostEqual(args[1], 10.0, places=2)

    def test_execute_direct(self) -> None:
        backend = MagicMock()
        handler = ExecutionHandler(backend)
        handler.execute_direct("AAPL", 10, "SELL")
        backend.submit_order.assert_called_with("AAPL", 10, "SELL")


if __name__ == "__main__":
    unittest.main()
