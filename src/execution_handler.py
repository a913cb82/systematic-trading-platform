from typing import Dict

from .base import ExecutionBackend


class ExecutionHandler:
    def __init__(self, backend: ExecutionBackend) -> None:
        self.backend = backend

    def rebalance(
        self,
        target_weights: Dict[int, float],
        reverse_ism: Dict[int, str],
        capital: float,
    ) -> None:
        """
        Converts target weights to orders based on current positions.
        """
        current_positions = self.backend.get_positions()
        # In a real system, we'd fetch real-time prices.
        # Here we'll use 1.0 as a simplified price or get it from backend.
        prices = self.backend.get_prices(list(reverse_ism.values()))

        trade_tolerance = 0.01
        for iid, weight in target_weights.items():
            ticker = reverse_ism[iid]
            price = prices.get(ticker, 100.0)  # Default to 100 for demo
            target_qty = (weight * capital) / price
            current_qty = current_positions.get(ticker, 0.0)

            diff = target_qty - current_qty
            if abs(diff) > trade_tolerance:  # Tolerance for small trades
                side = "BUY" if diff > 0 else "SELL"
                self.backend.submit_order(ticker, abs(diff), side)

    def execute_direct(self, ticker: str, quantity: float, side: str) -> bool:
        return self.backend.submit_order(ticker, quantity, side)
