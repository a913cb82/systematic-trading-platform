from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from src.core.types import OrderSide, Timeframe
from src.gateways.alpaca import AlpacaDataProvider, AlpacaExecutionBackend

# Constants to avoid magic values
AAPL_TICKER = "AAPL"
EXPECTED_CLOSE = 152.0
DEFAULT_QTY = 10


@patch("src.gateways.alpaca.StockHistoricalDataClient")
def test_alpaca_dataprovider_fetches_historical_bars_correctly(
    mock_client: Any,
) -> None:
    mock_instance = mock_client.return_value
    mock_bars = MagicMock()
    mock_df = pd.DataFrame(
        {
            "symbol": [AAPL_TICKER],
            "timestamp": [datetime(2025, 1, 1)],
            "open": [150.0],
            "high": [155.0],
            "low": [149.0],
            "close": [EXPECTED_CLOSE],
            "volume": [1000000],
        }
    ).set_index(["symbol", "timestamp"])
    mock_bars.df = mock_df
    mock_instance.get_stock_bars.return_value = mock_bars

    provider = AlpacaDataProvider("key", "secret")
    df = provider.fetch_bars(
        [AAPL_TICKER],
        datetime(2025, 1, 1),
        datetime(2025, 1, 2),
        timeframe=Timeframe.DAY,
    )

    assert not df.empty
    assert df.iloc[0]["ticker"] == AAPL_TICKER
    assert df.iloc[0]["close"] == EXPECTED_CLOSE


@patch("src.gateways.alpaca.TradingClient")
def test_alpaca_execution_backend_submits_market_orders(
    mock_trading_client: Any,
) -> None:
    mock_instance = mock_trading_client.return_value
    backend = AlpacaExecutionBackend("key", "secret")

    assert backend.submit_order(AAPL_TICKER, DEFAULT_QTY, OrderSide.BUY.value)
    mock_instance.submit_order.assert_called_once()

    mock_instance.submit_order.side_effect = Exception("API Error")
    assert not backend.submit_order(
        AAPL_TICKER, DEFAULT_QTY, OrderSide.SELL.value
    )


def test_alpaca_execution_backend_handles_position_retrieval_errors() -> None:
    with patch("src.gateways.alpaca.TradingClient") as mock_trade:
        inst = mock_trade.return_value
        inst.get_all_positions.side_effect = Exception("err")
        backend = AlpacaExecutionBackend("k", "s")
        assert backend.get_positions() == {}


def test_alpaca_execution_backend_handles_price_retrieval_errors() -> None:
    with patch("src.gateways.alpaca.StockHistoricalDataClient") as mock_hist:
        inst = mock_hist.return_value
        inst.get_stock_latest_quote.side_effect = Exception("err")
        backend = AlpacaExecutionBackend("k", "s")
        assert backend.get_prices([AAPL_TICKER]) == {}
