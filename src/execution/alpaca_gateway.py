import logging
from datetime import datetime
from typing import Callable, Dict, List, cast

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order, Position, TradeAccount
from alpaca.trading.requests import MarketOrderRequest

from ..common.config import config
from ..common.types import Trade
from ..data.ism import InternalSecurityMaster
from .broker_gateway import BrokerGateway

logger = logging.getLogger(__name__)


class AlpacaBrokerGateway(BrokerGateway):
    def __init__(self, ism: InternalSecurityMaster) -> None:
        self.ism = ism
        alpaca_cfg = config.execution.alpaca
        self.client = TradingClient(
            api_key=alpaca_cfg.key,
            secret_key=alpaca_cfg.secret,
            paper=alpaca_cfg.paper,
        )
        self._fill_callbacks: list[Callable[[Trade], None]] = []

    def connect(self) -> bool:
        try:
            account = cast(TradeAccount, self.client.get_account())
            logger.info(
                f"Connected to Alpaca. Account Status: {account.status}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def submit_order(
        self,
        internal_id: int,
        side: str,
        quantity: float,
        order_type: str = "MKT",
    ) -> str:
        # 1. Resolve ticker
        info = self.ism.get_symbol_info(internal_id, datetime.now())
        if not info:
            raise ValueError(
                f"Could not resolve ticker for internal_id {internal_id}"
            )

        ticker = info["ticker"]
        alpaca_side = (
            OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        )

        # 2. Prepare request
        if order_type.upper() == "MKT":
            req = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.GTC,
            )
        else:
            # Add more types if needed
            raise NotImplementedError(
                f"Order type {order_type} not implemented for Alpaca"
            )

        # 3. Submit
        try:
            order = cast(Order, self.client.submit_order(req))
            logger.info(f"Submitted Alpaca order: {order.id} for {ticker}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Alpaca order submission failed: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel Alpaca order {order_id}: {e}")
            return False

    def get_positions(self) -> Dict[int, float]:
        try:
            alpaca_positions = cast(
                List[Position], self.client.get_all_positions()
            )
            positions: Dict[int, float] = {}
            now = datetime.now()

            for pos in alpaca_positions:
                # Resolve ticker to internal_id
                iid = self.ism.get_internal_id_by_ticker(pos.symbol, now)
                if iid:
                    positions[iid] = float(pos.qty)
                else:
                    logger.warning(
                        f"Alpaca position in {pos.symbol} not found in ISM"
                    )

            return positions
        except Exception as e:
            logger.error(f"Failed to get Alpaca positions: {e}")
            return {}

    def subscribe_fills(self, callback: Callable[[Trade], None]) -> None:
        """
        In production, this would use Alpaca's TradeStream (WebSocket).
        For this PoC, we will rely on polling or the calling engine.
        """
        self._fill_callbacks.append(callback)
