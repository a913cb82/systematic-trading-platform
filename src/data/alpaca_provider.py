import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestBarRequest

from ..common.config import config
from ..common.types import Bar
from .live_provider import LiveDataProvider

logger = logging.getLogger(__name__)


class AlpacaLiveProvider(LiveDataProvider):
    """
    Real-world data provider using Alpaca Market Data API.
    Requires TRADING_SYSTEM_ALPACA_KEY and TRADING_SYSTEM_ALPACA_SECRET.
    """

    def __init__(self) -> None:
        self.api_key = config.get("alpaca.key")
        self.api_secret = config.get("alpaca.secret")
        self.internal_ids: List[int] = []
        self.symbol_map: Dict[str, int] = {}  # ticker -> internal_id
        self.callbacks: List[Callable[[Bar], None]] = []

        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca credentials missing in config.")
            self.client: Optional[StockHistoricalDataClient] = None
        else:
            self.client = StockHistoricalDataClient(
                self.api_key, self.api_secret
            )

    def connect(self) -> bool:
        if not self.client:
            logger.error("Cannot connect: Alpaca client not initialized.")
            return False
        logger.info("Connected to Alpaca Market Data.")
        return True

    def subscribe_bars(
        self, internal_ids: List[int], callback: Callable[[Bar], None]
    ) -> None:
        # In a real system, this would use Alpaca's WebSocket (StockDataStream)
        # For this implementation, we'll store the subscription and
        # let the LiveRunner pull via get_latest_bar or similar.
        self.internal_ids.extend(internal_ids)
        self.callbacks.append(callback)

        # We need a ticker map to query Alpaca
        # This assumes the user has registered them in ISM (handled in main.py)
        # For simplicity, we'll assume the caller provides a way to map them
        # or we'd query the ISM here.

    def get_latest_bar(self, internal_id: int) -> Optional[Bar]:
        """
        Polls Alpaca for the latest bar for a given internal_id.
        """
        if not self.client:
            return None

        # We need the ticker. In a real system, this provider would
        # have access to the ISM to resolve this.
        # For now, we'll try to find it in our local map.
        ticker = next(
            (t for t, i in self.symbol_map.items() if i == internal_id), None
        )
        if not ticker:
            logger.error(f"Ticker not found for internal_id {internal_id}")
            return None

        request_params = StockLatestBarRequest(symbol_or_symbols=ticker)
        latest_bars = self.client.get_stock_latest_bar(request_params)

        if ticker in latest_bars:
            alpaca_bar = latest_bars[ticker]
            return {
                "internal_id": internal_id,
                "timestamp": alpaca_bar.timestamp,
                "timestamp_knowledge": datetime.now(),
                "open": float(alpaca_bar.open),
                "high": float(alpaca_bar.high),
                "low": float(alpaca_bar.low),
                "close": float(alpaca_bar.close),
                "volume": float(alpaca_bar.volume),
            }
        return None

    def disconnect(self) -> None:
        logger.info("Disconnected from Alpaca.")

    def write_bars(self, data: List[Bar], fill_gaps: bool = False) -> None:
        pass

    def get_bars(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        adjustment: str,
        as_of: Optional[datetime] = None,
    ) -> List[Bar]:
        return []

    def get_universe(self, date: datetime) -> List[int]:
        return self.internal_ids

    def get_returns(
        self,
        internal_ids: List[int],
        date_range: tuple[datetime, datetime],
        type: str = "RAW",
        as_of: Optional[datetime] = None,
        risk_model: Optional[Any] = None,
    ) -> Any:
        return []
