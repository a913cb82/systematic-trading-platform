from enum import Enum
from typing import Dict, List

from .base import ExecutionBackend


class OrderState(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Order:
    def __init__(self, ticker: str, quantity: float, side: str):
        self.ticker = ticker
        self.quantity = quantity
        self.filled_qty = 0.0
        self.side = side
        self.state = OrderState.PENDING

    def update(self, fill_qty: float) -> None:
        self.filled_qty += fill_qty
        if self.filled_qty >= self.quantity:
            self.state = OrderState.FILLED
        elif self.filled_qty > 0:
            self.state = OrderState.PARTIAL


class ExecutionHandler:
    def __init__(self, backend: ExecutionBackend) -> None:
        self.backend = backend
        self.orders: List[Order] = []

    def vwap_execute(
        self, ticker: str, total_qty: float, side: str, slices: int = 5
    ) -> None:
        """
        Simple VWAP/TWAP slicing logic (simulated).
        """
        qty_per_slice = total_qty / slices
        order = Order(ticker, total_qty, side)
        order.state = OrderState.SUBMITTED
        self.orders.append(order)

        for _ in range(slices):
            success = self.backend.submit_order(ticker, qty_per_slice, side)
            if success:
                order.update(qty_per_slice)
            else:
                order.state = OrderState.REJECTED
                break

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
        prices = self.backend.get_prices(list(reverse_ism.values()))

        trade_tolerance = 0.01
        for iid, weight in target_weights.items():
            ticker = reverse_ism[iid]
            price = prices.get(ticker, 100.0)
            target_qty = (weight * capital) / price
            current_qty = current_positions.get(ticker, 0.0)

            diff = target_qty - current_qty
            if abs(diff) > trade_tolerance:
                side = "BUY" if diff > 0 else "SELL"
                self.vwap_execute(ticker, abs(diff), side)

    def execute_direct(self, ticker: str, quantity: float, side: str) -> bool:
        return self.backend.submit_order(ticker, quantity, side)


class TCAEngine:
    """
    Post-Trade Analysis & Attribution.
    """

    @staticmethod
    def calculate_slippage(
        arrival_price: float, execution_price: float, side: str
    ) -> float:
        """
        Calculates slippage in basis points.
        """
        if arrival_price == 0:
            return 0.0
        sign = 1 if side == "BUY" else -1
        return sign * (execution_price - arrival_price) / arrival_price * 10000


class FIXEngine:
    """
    Minimal FIX Protocol Connectivity skeleton.
    """

    def __init__(self, target_comp_id: str):
        self.target_comp_id = target_comp_id
        self.connected = False

    def logon(self) -> bool:
        self.connected = True
        return True

    def send_order(self, ticker: str, qty: float, side: str) -> str:
        # Mock FIX message 35=D
        return f"FIX_ORDER_ID_{ticker}_{qty}"
