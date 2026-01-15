import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, cast

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar as AlpacaBar

from ..common.config import config
from ..common.types import Bar
from ..data.ism import InternalSecurityMaster
from .live_provider import LiveDataProvider

logger = logging.getLogger(__name__)


class AlpacaLiveProvider(LiveDataProvider):
    """
    Real-world data provider using Alpaca Market Data API (WebSockets).
    """

    def __init__(self, ism: InternalSecurityMaster) -> None:
        self.ism = ism
        alpaca_cfg = config.execution.alpaca
        self.api_key = alpaca_cfg.key
        self.api_secret = alpaca_cfg.secret

        self.stream = StockDataStream(self.api_key, self.api_secret)
        self.historical_client = StockHistoricalDataClient(
            self.api_key, self.api_secret
        )

        self.internal_ids: List[int] = []
        self._callbacks: List[Callable[[Bar], None]] = []
        self._ticker_to_id: Dict[str, int] = {}
        self._stream_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        try:
            # We don't "connect" here in a blocking way,
            # we start the stream in a separate thread later.
            logger.info("AlpacaLiveProvider initialized.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AlpacaLiveProvider: {e}")
            return False

    def subscribe_bars(
        self, internal_ids: List[int], callback: Callable[[Bar], None]
    ) -> None:
        self.internal_ids.extend(internal_ids)
        self._callbacks.append(callback)

        now = datetime.now()
        tickers = []
        for iid in internal_ids:
            info = self.ism.get_symbol_info(iid, now)
            if info:
                ticker = info["ticker"]
                tickers.append(ticker)
                self._ticker_to_id[ticker] = iid
            else:
                logger.warning(f"Could not resolve ticker for iid {iid}")

        if not tickers:
            return

        # Define handler
        async def bar_handler(data: Union[AlpacaBar, Dict[Any, Any]]) -> None:
            if isinstance(data, dict):
                symbol = cast(str, data.get("symbol"))
                timestamp = cast(datetime, data.get("timestamp"))
                open_p = data.get("open")
                high_p = data.get("high")
                low_p = data.get("low")
                close_p = data.get("close")
                volume_p = data.get("volume")
            else:
                symbol = data.symbol
                timestamp = data.timestamp
                open_p = data.open
                high_p = data.high
                low_p = data.low
                close_p = data.close
                volume_p = data.volume

            iid = self._ticker_to_id.get(symbol) if symbol else None
            if iid is None:
                return

            normalized_bar: Bar = {
                "internal_id": iid,
                "timestamp": timestamp,
                "timestamp_knowledge": datetime.now(),
                "open": float(open_p) if open_p is not None else 0.0,
                "high": float(high_p) if high_p is not None else 0.0,
                "low": float(low_p) if low_p is not None else 0.0,
                "close": float(close_p) if close_p is not None else 0.0,
                "volume": float(volume_p) if volume_p is not None else 0.0,
            }

            for cb in self._callbacks:
                cb(normalized_bar)

        self.stream.subscribe_bars(bar_handler, *tickers)

        # Start streaming in a background thread if not already running
        if self._stream_thread is None:
            self._stream_thread = threading.Thread(
                target=self.stream.run, daemon=True
            )
            self._stream_thread.start()
            logger.info(f"Started Alpaca WebSocket stream for {tickers}")

    def get_latest_bar(self, internal_id: int) -> Optional[Bar]:
        return None

    def disconnect(self) -> None:
        if self.stream:
            logger.info("Alpaca stream disconnect requested.")

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
