import time
import unittest
from unittest.mock import MagicMock

from src.core.execution_handler import ExecutionHandler, OrderState


class TestExecutionHandler(unittest.TestCase):
    def test_rebalance_basic(self) -> None:
        backend = MagicMock()
        backend.get_positions.return_value = {"AAPL": 10.0}
        backend.get_prices.return_value = {"AAPL": 100.0}

        handler = ExecutionHandler(backend)
        # 50% of 10000 = 5000 / 100 = 50 shares. Diff = +40
        # Goal positions passed directly now
        goal_positions = {"AAPL": 50.0}

        handler.rebalance(goal_positions, interval=0)

        # Wait for thread to finish
        max_wait = 1.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if handler.orders and handler.orders[0].state == OrderState.FILLED:
                break
            time.sleep(0.01)

        # Default vwap_execute uses 5 slices. 40 / 5 = 8.0
        backend.submit_order.assert_called_with("AAPL", 8.0, "BUY")
        self.assertEqual(backend.submit_order.call_count, 5)

    def test_rebalance_multi_asset(self) -> None:
        backend = MagicMock()
        backend.get_positions.return_value = {"AAPL": 100.0, "MSFT": 50.0}
        backend.get_prices.return_value = {"AAPL": 100.0, "MSFT": 200.0}

        handler = ExecutionHandler(backend)
        # Target: AAPL 100 shares (no change), MSFT 60 shares (+10)
        goal_positions = {"AAPL": 100.0, "MSFT": 60.0}

        handler.rebalance(goal_positions, interval=0)

        # Wait for threads to finish
        max_wait = 1.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if all(o.state == OrderState.FILLED for o in handler.orders):
                break
            time.sleep(0.01)

        # Should only call submit_order for MSFT, 5 times (slices)
        self.assertEqual(backend.submit_order.call_count, 5)
        args, _ = backend.submit_order.call_args
        self.assertEqual(args[0], "MSFT")
        self.assertAlmostEqual(args[1], 2.0, places=2)  # 10 / 5 = 2.0

    def test_execute_direct(self) -> None:
        backend = MagicMock()
        handler = ExecutionHandler(backend)
        handler.execute_direct("AAPL", 10, "SELL")
        backend.submit_order.assert_called_with("AAPL", 10, "SELL")


if __name__ == "__main__":
    unittest.main()
