import uuid
from datetime import datetime
from typing import Dict, Optional

from ..common.base import ExecutionEngine, MarketDataProvider
from ..common.types import Order, OrderStatus, Trade
from ..portfolio.publisher import TargetWeightPublisher
from .safety import SafetyLayer

RECONCILIATION_TOLERANCE = 1e-5


class OrderManagementSystem:
    def __init__(
        self,
        weight_publisher: TargetWeightPublisher,
        execution_engine: ExecutionEngine,
        market_data: Optional[MarketDataProvider] = None,
        safety_layer: Optional[SafetyLayer] = None,
        capital: float = 1_000_000.0,
    ):
        self.weight_publisher = weight_publisher
        self.execution_engine = execution_engine
        self.market_data = market_data
        self.safety_layer = safety_layer or SafetyLayer()
        self.capital = capital
        self.current_targets: Dict[int, float] = {}
        self.internal_positions: Dict[int, float] = {}  # internal_id -> shares
        self.active_orders: Dict[str, Order] = {}  # order_id -> Order

        # Subscribe to target weights
        self.weight_publisher.subscribe_target_weights(self.on_target_weights)

        # Subscribe to fills from the execution engine
        self.execution_engine.subscribe_fills(self.on_fill)

    def on_fill(self, fill: Trade) -> None:
        """
        Updates internal positions and order states based on realized trades.
        """
        iid = fill["internal_id"]
        shares = (
            fill["quantity"] if fill["side"] == "BUY" else -fill["quantity"]
        )
        self.internal_positions[iid] = (
            self.internal_positions.get(iid, 0.0) + shares
        )

        # Update order status if we can match it
        # (in a real system we'd have order_id in Trade)
        # For this simulated environment, we'll find the first matching
        # open order
        for _, order in self.active_orders.items():
            if (
                order["internal_id"] == iid
                and order["side"] == fill["side"]
                and order["status"]
                in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
            ):
                order["filled_quantity"] += fill["quantity"]
                if order["filled_quantity"] >= order["quantity"]:
                    order["status"] = OrderStatus.FILLED
                else:
                    order["status"] = OrderStatus.PARTIALLY_FILLED
                break

    def reconcile(self, broker_positions: Dict[int, float]):
        """
        Compares internal positions with broker state and logs discrepancies.
        """
        all_ids = set(self.internal_positions.keys()) | set(
            broker_positions.keys()
        )
        discrepancies = []

        for iid in all_ids:
            internal = self.internal_positions.get(iid, 0.0)
            broker = broker_positions.get(iid, 0.0)
            if abs(internal - broker) > RECONCILIATION_TOLERANCE:
                discrepancies.append(
                    {
                        "internal_id": iid,
                        "internal": internal,
                        "broker": broker,
                        "diff": internal - broker,
                    }
                )

        if discrepancies:
            print(f"RECONCILIATION DISCREPANCY: {discrepancies}")
            # In production, we would force-update internal positions
            # to broker state
            for d in discrepancies:
                self.internal_positions[d["internal_id"]] = d["broker"]  # type: ignore
        else:
            print(
                "RECONCILIATION SUCCESS: Internal and Broker positions match."
            )

    def on_target_weights(
        self, timestamp: datetime, weights: Dict[int, float]
    ) -> None:
        """
        Callback triggered when new target weights are available.
        Calculates required orders and passes them to execution.
        """
        is_valid, reason = self.safety_layer.validate_weights(
            weights, market_data=self.market_data, capital=self.capital
        )
        if not is_valid:
            print(f"Safety check failed: {reason}")
            return

        # Calculate differences before updating current_targets
        for iid, weight in weights.items():
            prev_weight = self.current_targets.get(iid, 0.0)
            if weight != prev_weight:
                order_id = str(uuid.uuid4())
                side = "BUY" if weight > prev_weight else "SELL"

                new_order: Order = {
                    "order_id": order_id,
                    "internal_id": iid,
                    "side": side,
                    "quantity": abs(weight - prev_weight),
                    "filled_quantity": 0.0,
                    "status": OrderStatus.SUBMITTED,
                    "timestamp": timestamp,
                }
                self.active_orders[order_id] = new_order

        self.current_targets = weights
        self.execution_engine.execute(timestamp, weights)
