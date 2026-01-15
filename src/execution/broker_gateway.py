from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict

from ..common.types import Trade


class BrokerGateway(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def submit_order(
        self,
        internal_id: int,
        side: str,
        quantity: float,
        order_type: str = "MKT",
    ) -> str:
        """Returns an order_id"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_positions(self) -> Dict[int, float]:
        pass

    @abstractmethod
    def subscribe_fills(self, callback: Callable[[Trade], None]) -> None:
        pass


class MockBrokerGateway(BrokerGateway):
    """
    Simulates a broker gateway for testing.
    """

    def __init__(self, execution_engine: Any) -> None:
        self.execution_engine = execution_engine
        self.positions: Dict[int, float] = {}
        self.order_id_counter = 0

    def connect(self) -> bool:
        return True

    def submit_order(
        self,
        internal_id: int,
        side: str,
        quantity: float,
        order_type: str = "MKT",
    ) -> str:
        self.order_id_counter += 1
        order_id = f"ORDER_{self.order_id_counter}"

        # In a real system, this would be asynchronous.
        # Here we just tell the engine to execute it immediately (simulated).
        # Note: execute usually takes weights, but here we are sending an
        # order for shares.
        # This highlights the gap between Target Weights and Orders.

        # For simplicity, we'll just mock a fill
        fill = Trade(
            internal_id=internal_id,
            side=side,
            quantity=quantity,
            price=100.0,  # Mock price
            fees=0.0,
            venue="MOCK_EXCHANGE",
            timestamp=datetime.now(),
        )
        self.execution_engine.report_fill(fill)

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_positions(self) -> Dict[int, float]:
        return self.positions

    def subscribe_fills(self, callback: Callable[[Trade], None]) -> None:
        self.execution_engine.subscribe_fills(callback)
