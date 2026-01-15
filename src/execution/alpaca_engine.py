import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List

from ..common.base import ExecutionEngine, MarketDataProvider
from ..common.types import Trade
from .alpaca_gateway import AlpacaBrokerGateway
from .order_generator import OrderGenerator

logger = logging.getLogger(__name__)


class AlpacaExecutionEngine(ExecutionEngine):
    def __init__(
        self,
        gateway: AlpacaBrokerGateway,
        market_data: MarketDataProvider,
        capital: float = 100000.0,
    ) -> None:
        self.gateway = gateway
        self.market_data = market_data
        self.capital = capital
        self.order_generator = OrderGenerator()
        self._subscribers: List[Callable[[Trade], None]] = []

    def execute(
        self, timestamp: datetime, target_weights: Dict[int, float]
    ) -> None:
        """
        Calculates required orders from weights and executes them via Alpaca.
        """
        # 1. Get current positions
        current_positions = self.gateway.get_positions()

        # 2. Get current prices for conversion
        # Use latest bars from market data engine
        current_prices: Dict[int, float] = {}
        for iid in target_weights.keys():
            bars = self.market_data.get_bars(
                [iid],
                timestamp - timedelta(days=1),
                timestamp,
                adjustment="RAW",
            )
            if bars:
                current_prices[iid] = bars[-1]["close"]

        # 3. Generate orders
        orders = self.order_generator.generate_orders(
            target_weights, current_positions, self.capital, current_prices
        )

        # 4. Submit orders
        for order in orders:
            try:
                self.gateway.submit_order(
                    order["internal_id"], order["side"], order["quantity"]
                )
            except Exception as e:
                logger.error(
                    f"Failed to submit order for {order['internal_id']}: {e}"
                )

    def report_fill(self, fill: Trade) -> None:
        # In Alpaca, fills come from the gateway (TradeStream)
        # We would propagate them here.
        for callback in self._subscribers:
            callback(fill)

    def get_fills(
        self, start_time: datetime, end_time: datetime
    ) -> List[Trade]:
        # Would query Alpaca historical orders/trades
        return []

    def subscribe_fills(self, on_fill: Callable[[Trade], None]) -> None:
        self._subscribers.append(on_fill)
        self.gateway.subscribe_fills(on_fill)
