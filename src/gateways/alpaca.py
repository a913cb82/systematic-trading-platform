from datetime import datetime
from typing import Dict, List, cast

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .base import DataProvider, ExecutionBackend


class AlpacaDataProvider(DataProvider):
    def __init__(self, api_key: str, api_secret: str) -> None:
        self.client = StockHistoricalDataClient(api_key, api_secret)

    def fetch_bars(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = self.client.get_stock_bars(request_params)
        df = getattr(bars, "df").reset_index()
        df = df.rename(columns={"symbol": "ticker"})
        return df[
            ["ticker", "timestamp", "open", "high", "low", "close", "volume"]
        ]

    def fetch_corporate_actions(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        # Alpaca doesn't have a simple historical CA endpoint in the basic SDK.
        return pd.DataFrame(columns=["ticker", "ex_date", "type", "ratio"])

    def fetch_events(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch fundamental or other irregular events."""
        return pd.DataFrame(
            columns=["ticker", "timestamp", "event_type", "value"]
        )


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
            time_in_force=TimeInForce.GTC,
        )
        try:
            self.client.submit_order(order_data)
            return True
        except Exception as e:
            print(f"Alpaca Order Error: {e}")
            return False

    def get_positions(self) -> Dict[str, float]:
        positions = self.client.get_all_positions()
        return {
            cast(str, getattr(p, "symbol")): float(getattr(p, "qty"))
            for p in positions
        }

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            return {}
        from alpaca.common.exceptions import APIError
        from alpaca.data.requests import StockLatestQuoteRequest

        request_params = StockLatestQuoteRequest(symbol_or_symbols=tickers)
        try:
            quotes = self.data_client.get_stock_latest_quote(request_params)
            return {symbol: float(q.ask_price) for symbol, q in quotes.items()}
        except APIError:
            return {}
        except Exception as e:
            print(f"Alpaca Price Error: {e}")
            return {}
