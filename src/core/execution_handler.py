import threading
import time
from typing import Dict, List, Optional

from src.core.types import ChildOrder, Order, OrderSide, OrderState
from src.gateways.base import ExecutionBackend


class ExecutionHandler:
    """
    Live Execution Manager.
    - Manages exchange connectivity and order state.
    - Centralized background thread manages child order slicing.
    """

    def __init__(self, backend: ExecutionBackend) -> None:
        self.backend = backend
        self.orders: List[Order] = []
        self._queue: List[ChildOrder] = []
        self._lock = threading.Lock()

        # Start centralized execution worker
        self._worker = threading.Thread(
            target=self._execution_loop, daemon=True
        )
        self._worker.start()

    def _execution_loop(self) -> None:
        """Main loop for submitting sliced child orders."""
        while True:
            now = time.time()
            with self._lock:
                to_fire = [c for c in self._queue if c.scheduled_at <= now]
                self._queue = [c for c in self._queue if c.scheduled_at > now]

            for child in to_fire:
                if not child.parent.state.is_active:
                    continue

                success = self.backend.submit_order(
                    child.parent.ticker,
                    child.quantity,
                    child.parent.side.value,
                )
                if success:
                    child.parent.update(child.quantity)
                else:
                    child.parent.state = OrderState.REJECTED

            time.sleep(0.1)

    def vwap_execute(
        self,
        ticker: str,
        total_qty: float,
        side: OrderSide,
        slices: int = 5,
        interval: float = 1.0,
    ) -> Order:
        """
        Spaced-out execution (TWAP-style).
        Enqueues child orders into the centralized background worker.
        """
        order = Order(ticker, total_qty, side)
        order.state = OrderState.SUBMITTED
        self.orders.append(order)

        qty_per_slice = total_qty / slices
        now = time.time()

        with self._lock:
            for i in range(slices):
                self._queue.append(
                    ChildOrder(
                        parent=order,
                        quantity=qty_per_slice,
                        scheduled_at=now + (i * interval),
                    )
                )
            self._queue.sort(key=lambda x: x.scheduled_at)

        return order

    def rebalance(
        self, goal_positions: Dict[str, float], interval: float = 1.0
    ) -> List[Order]:
        """
        Executes trades to reach goal positions (share counts).
        Cancels any existing active orders for the tickers being updated.
        """
        # 1. Cancel active orders for tickers we are about to rebalance
        all_tickers = set(goal_positions.keys())
        for o in self.orders:
            if o.ticker in all_tickers and o.state.is_active:
                self.cancel_order(o.order_id)

        # 2. Source of Truth: Get positions from backend
        current_positions = self.backend.get_positions()
        trade_tolerance = 0.1

        # Combine goal tickers and existing position tickers
        all_tickers |= set(current_positions.keys())

        new_orders = []
        for ticker in all_tickers:
            target = goal_positions.get(ticker, 0.0)
            current = current_positions.get(ticker, 0.0)
            diff = target - current

            if abs(diff) > trade_tolerance:
                order = self.vwap_execute(
                    ticker,
                    abs(diff),
                    OrderSide.BUY if diff > 0 else OrderSide.SELL,
                    interval=interval,
                )
                new_orders.append(order)
        return new_orders

    def execute_direct(
        self, ticker: str, quantity: float, side: OrderSide
    ) -> bool:
        return self.backend.submit_order(ticker, quantity, side.value)

    def cancel_order(self, order_id: int) -> bool:
        """Attempts to cancel a parent order (stops further slicing)."""
        order = self.get_order(order_id)
        if order and order.state.is_active:
            order.state = OrderState.CANCELLED
            return True
        return False

    def get_order(self, order_id: int) -> Optional[Order]:
        """Retrieves a parent order by ID."""
        return next((o for o in self.orders if o.order_id == order_id), None)


class TCAEngine:
    @staticmethod
    def calculate_slippage(
        arrival_price: float, execution_price: float, side: OrderSide
    ) -> float:
        if arrival_price == 0:
            return 0.0
        sign = 1 if side == OrderSide.BUY else -1
        return sign * (execution_price - arrival_price) / arrival_price * 10000


class FIXEngine:
    def __init__(self, target_comp_id: str):
        self.target_comp_id = target_comp_id
        self.connected = False

    def logon(self) -> bool:
        self.connected = True
        return True

    def send_order(self, ticker: str, qty: float, side: OrderSide) -> str:
        return f"FIX_ORDER_ID_{ticker}_{qty}"
