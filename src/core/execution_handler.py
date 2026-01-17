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
    """
    Live Execution Manager.
    - Manages exchange connectivity and order state.
    - Pass-through for real-time data ingestion into DataPlatform.
    """

    def __init__(
        self,
        backend: ExecutionBackend,
        data_platform: Optional[DataPlatform] = None,
    ) -> None:
        self.backend = backend
        self.data = data_platform
        self.orders: List[Order] = []

    def on_bar(self, bar: Bar) -> None:
        """Pass-through ingestion to DataPlatform."""
        if self.data:
            self.data.add_bars([bar])

    def vwap_execute(
        self, ticker: str, total_qty: float, side: str, slices: int = 5
    ) -> None:
        """Simple VWAP/TWAP slicing logic."""
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

    def rebalance(self, goal_positions: Dict[str, float]) -> None:
        """Executes trades to reach goal positions (share counts)."""
        current_positions = self.backend.get_positions()
        trade_tolerance = 0.1
        all_tickers = set(current_positions.keys()) | set(
            goal_positions.keys()
        )

        for ticker in all_tickers:
            target = goal_positions.get(ticker, 0.0)
            current = current_positions.get(ticker, 0.0)
            diff = target - current

            if abs(diff) > trade_tolerance:
                self.vwap_execute(
                    ticker, abs(diff), "BUY" if diff > 0 else "SELL"
                )

    def execute_direct(self, ticker: str, quantity: float, side: str) -> bool:
        return self.backend.submit_order(ticker, quantity, side)


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
