import time
import unittest
from unittest.mock import MagicMock

from src.core.execution_handler import ExecutionHandler, OrderState


class TestExecutionAsync(unittest.TestCase):
    def test_order_cancellation_stops_slicing(self) -> None:
        backend = MagicMock()
        backend.submit_order.return_value = True
        handler = ExecutionHandler(backend)

        # 10 slices, 0.1s apart. Total time 1s.
        order = handler.vwap_execute(
            "AAPL", 100, "BUY", slices=10, interval=0.1
        )

        # Wait for worker to start and process first slice
        max_wait = 1.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if backend.submit_order.call_count >= 1:
                break
            time.sleep(0.01)

        self.assertTrue(backend.submit_order.call_count >= 1)

        # Cancel order
        handler.cancel_order(order.order_id)
        self.assertEqual(order.state, OrderState.CANCELLED)

        # Wait long enough for more slices to have fired if it wasn't cancelled
        time.sleep(0.3)

        # Should still be 1 or 2 if race condition, but not 10
        max_allowed = 2
        self.assertTrue(backend.submit_order.call_count <= max_allowed)

    def test_rejection_fails_parent(self) -> None:
        backend = MagicMock()
        # Fail the second slice
        backend.submit_order.side_effect = [True, False]
        handler = ExecutionHandler(backend)

        order = handler.vwap_execute(
            "AAPL", 100, "BUY", slices=2, interval=0.01
        )

        # Wait for both slices to process (or worker to stop)
        max_wait = 1.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if order.state == OrderState.REJECTED:
                break
            time.sleep(0.01)

        self.assertEqual(order.state, OrderState.REJECTED)
        # Should have only called twice (stopped after failure)
        self.assertEqual(backend.submit_order.call_count, 2)


if __name__ == "__main__":
    unittest.main()
