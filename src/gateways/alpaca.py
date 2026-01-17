import logging
from datetime import datetime
from typing import Callable, Dict, List, cast

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar as AlpacaBar
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.core.types import Bar, Timeframe

from .base import (
    BarProvider,
    CorporateActionProvider,
    EventProvider,
    ExecutionBackend,
    StreamProvider,
)

logger = logging.getLogger(__name__)


class AlpacaDataProvider(BarProvider, CorporateActionProvider, EventProvider):
    def __init__(self, api_key: str, api_secret: str) -> None:
        self.client = StockHistoricalDataClient(api_key, api_secret)

    def _map_timeframe(self, timeframe: Timeframe) -> AlpacaTimeFrame:
        mapping = {
            Timeframe.MINUTE: (1, TimeFrameUnit.Minute),
            Timeframe.MIN_5: (5, TimeFrameUnit.Minute),
            Timeframe.MIN_15: (15, TimeFrameUnit.Minute),
            Timeframe.MIN_30: (30, TimeFrameUnit.Minute),
            Timeframe.HOUR: (1, TimeFrameUnit.Hour),
            Timeframe.DAY: (1, TimeFrameUnit.Day),
        }
        val, unit = mapping.get(timeframe, (1, TimeFrameUnit.Minute))
        return AlpacaTimeFrame(val, unit)

    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = Timeframe.DAY,
    ) -> pd.DataFrame:
        alpaca_tf = self._map_timeframe(timeframe)
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=alpaca_tf,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        try:
            bars = self.client.get_stock_bars(request_params)
            df = getattr(bars, "df")
            if df.empty:
                return pd.DataFrame(
                    columns=[
                        "ticker",
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                )

            df = df.reset_index()
            df = df.rename(columns={"symbol": "ticker"})
            return df[
                [
                    "ticker",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]
        except Exception as e:
            logger.error(f"Alpaca Fetch Bars Error: {e}")
            return pd.DataFrame()

    def fetch_corporate_actions(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ticker", "ex_date", "type", "value"])

    def fetch_events(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["ticker", "timestamp", "event_type", "value"]
        )


class AlpacaRealtimeClient(StreamProvider):
    def __init__(self, api_key: str, api_secret: str) -> None:
        self.stream = StockDataStream(api_key, api_secret, feed=DataFeed.IEX)

    async def _handle_bar(
        self, data: AlpacaBar, handler: Callable[[Bar], None]
    ) -> None:
        internal_bar = Bar(
            internal_id=0,
            timestamp=data.timestamp.replace(tzinfo=None),
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            timeframe=Timeframe.MINUTE,
        )
        # Using dict access to avoid type errors on dataclass if needed,
        # but DataPlatform looks for it.
        setattr(internal_bar, "_ticker", data.symbol)
        handler(internal_bar)

    def subscribe(
        self, tickers: List[str], handler: Callable[[Bar], None]
    ) -> None:
        async def bar_callback(data: AlpacaBar) -> None:
            await self._handle_bar(data, handler)

        self.stream.subscribe_bars(bar_callback, *tickers)  # type: ignore[arg-type]

    def run(self) -> None:
        self.stream.run()


class AlpacaExecutionBackend(ExecutionBackend):
    def __init__(
        self, api_key: str, api_secret: str, paper: bool = True
    ) -> None:
        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)

    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=abs(quantity),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            self.client.submit_order(order_data)
            return True
        except Exception as e:
            logger.error(f"Alpaca Order Error: {e}")
            return False

    def get_positions(self) -> Dict[str, float]:
        try:
            positions = self.client.get_all_positions()
            return {
                cast(str, getattr(p, "symbol")): float(getattr(p, "qty"))
                for p in positions
            }
        except Exception as e:
            logger.error(f"Alpaca Get Positions Error: {e}")
            return {}

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            return {}

        request_params = StockLatestQuoteRequest(symbol_or_symbols=tickers)
        try:
            quotes = self.data_client.get_stock_latest_quote(request_params)
            return {symbol: float(q.ask_price) for symbol, q in quotes.items()}
        except APIError as e:
            logger.error(f"Alpaca API Error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Alpaca Price Error: {e}")
            return {}
