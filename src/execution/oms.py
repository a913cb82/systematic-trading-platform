from datetime import datetime
from typing import Dict, Optional

from ..common.base import ExecutionEngine
from ..common.types import Trade
from ..portfolio.publisher import TargetWeightPublisher
from .safety import SafetyLayer

RECONCILIATION_TOLERANCE = 1e-5


class OrderManagementSystem:
    def __init__(
        self,
        weight_publisher: TargetWeightPublisher,
        execution_engine: ExecutionEngine,
        safety_layer: Optional[SafetyLayer] = None,
    ):
        self.weight_publisher = weight_publisher
        self.execution_engine = execution_engine
        self.safety_layer = safety_layer or SafetyLayer()
        self.current_targets: Dict[int, float] = {}
        self.internal_positions: Dict[int, float] = {}  # internal_id -> shares

        # Subscribe to target weights
        self.weight_publisher.subscribe_target_weights(self.on_target_weights)

        # In a real system, we'd also subscribe to fills from the
        # execution engine
        self.execution_engine.subscribe_fills(self.on_fill)

    def on_fill(self, fill: Trade) -> None:
        """
        Updates internal positions based on realized trades.
        """
        iid = fill["internal_id"]
        shares = (
            fill["quantity"] if fill["side"] == "BUY" else -fill["quantity"]
        )
        self.internal_positions[iid] = (
            self.internal_positions.get(iid, 0.0) + shares
        )

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
        Passes them to the execution engine after safety checks.
        """
        is_valid, reason = self.safety_layer.validate_weights(weights)
        if not is_valid:
            print(f"Safety check failed: {reason}")
            # In production, this might trigger an alert or a partial execution
            return

        self.current_targets = weights
        self.execution_engine.execute(timestamp, weights)
