from datetime import datetime
from typing import Dict, List, Optional
from ..common.types import Trade
from ..common.base import ExecutionEngine
from ..portfolio.publisher import TargetWeightPublisher

from .safety import SafetyLayer

class OrderManagementSystem:
    def __init__(self, weight_publisher: TargetWeightPublisher, execution_engine: ExecutionEngine, safety_layer: Optional[SafetyLayer] = None):
        self.weight_publisher = weight_publisher
        self.execution_engine = execution_engine
        self.safety_layer = safety_layer or SafetyLayer()
        self.current_targets: Dict[int, float] = {}
        
        # Subscribe to target weights
        self.weight_publisher.subscribe_target_weights(self.on_target_weights)

    def on_target_weights(self, timestamp: datetime, weights: Dict[int, float]) -> None:
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
