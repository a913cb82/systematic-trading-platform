import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from src.core.data_platform import Bar, DataPlatform
from src.gateways.base import ExecutionBackend


class OrderState(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_active(self) -> bool:
        return self in (
            OrderState.PENDING,
            OrderState.SUBMITTED,
            OrderState.PARTIAL,
        )


class Order:
    _id_counter = 1
    _id_lock = threading.Lock()

    def __init__(self, ticker: str, quantity: float, side: str):
        with Order._id_lock:
            self.order_id = Order._id_counter
            Order._id_counter += 1
        self.ticker = ticker
        self.quantity = quantity
        self.filled_qty = 0.0
        self.side = side
        self.state = OrderState.PENDING
        self.timestamp = datetime.now()

    def update(self, fill_qty: float) -> None:
        self.filled_qty += fill_qty
        self.state = (
            OrderState.FILLED
            if self.filled_qty >= self.quantity
            else OrderState.PARTIAL
        )


@dataclass
class _ChildOrder:
    parent: "Order"
    quantity: float
    scheduled_at: float


class ExecutionHandler:
    """
    Live Execution Manager.
    - Manages exchange connectivity and order state.
    - Pass-through for real-time data ingestion into DataPlatform.
    - Centralized background thread manages child order slicing.
    """

    def __init__(
        self,
        backend: ExecutionBackend,
        data_platform: Optional[DataPlatform] = None,
    ) -> None:
        self.backend = backend
        self.data = data_platform
        self.orders: List[Order] = []
        self._queue: List[_ChildOrder] = []
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
                    child.parent.ticker, child.quantity, child.parent.side
                )
                if success:
                    child.parent.update(child.quantity)
                else:
                    child.parent.state = OrderState.REJECTED

            time.sleep(0.1)

    def on_bar(self, bar: Bar) -> None:
        """Pass-through ingestion to DataPlatform."""
        if self.data:
            self.data.add_bars([bar])

    def vwap_execute(
        self,
        ticker: str,
        total_qty: float,
        side: str,
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
                    _ChildOrder(
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
                    "BUY" if diff > 0 else "SELL",
                    interval=interval,
                )
                new_orders.append(order)
        return new_orders

    def execute_direct(self, ticker: str, quantity: float, side: str) -> bool:
        return self.backend.submit_order(ticker, quantity, side)

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
        arrival_price: float, execution_price: float, side: str
    ) -> float:
        if arrival_price == 0:
            return 0.0
        sign = 1 if side == "BUY" else -1
        return sign * (execution_price - arrival_price) / arrival_price * 10000


class FIXEngine:
    def __init__(self, target_comp_id: str):
        self.target_comp_id = target_comp_id
        self.connected = False

    def logon(self) -> bool:
        self.connected = True
        return True

    def send_order(self, ticker: str, qty: float, side: str) -> str:
        return f"FIX_ORDER_ID_{ticker}_{qty}"
