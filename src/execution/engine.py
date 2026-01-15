from datetime import datetime
from typing import Dict, List, Callable, Any
from ..common.base import ExecutionEngine
from ..common.types import Trade

from .algos import ExecutionAlgorithm

class SimulatedExecutionEngine(ExecutionEngine):
    def __init__(self, algo: ExecutionAlgorithm):
        self.algo = algo
        self.trades: List[Trade] = []
        self._subscribers: List[Callable[[Trade], None]] = []

    def execute(self, timestamp: datetime, target_weights: Dict[int, float]) -> None:
        """
        Simulate execution using the provided algorithm.
        """
        fills = self.algo.simulate_fills(timestamp, target_weights)
        for fill in fills:
            self.report_fill(fill)

    def report_fill(self, fill: Trade) -> None:
        self.trades.append(fill)
        for callback in self._subscribers:
            callback(fill)

    def get_fills(self, start_time: datetime, end_time: datetime) -> List[Trade]:
        return [
            t for t in self.trades 
            if start_time <= t['timestamp'] <= end_time
        ]

    def subscribe_fills(self, on_fill: Callable[[Trade], None]) -> None:
        self._subscribers.append(on_fill)