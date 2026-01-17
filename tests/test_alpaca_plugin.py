import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.core.types import Timeframe
from src.gateways.alpaca import AlpacaDataProvider, AlpacaExecutionBackend


class TestAlpacaPlugin(unittest.TestCase):
    def setUp(self) -> None:
        self.api_key = "test_key"
        self.api_secret = "test_secret"

    @patch("src.gateways.alpaca.StockHistoricalDataClient")
    def test_data_provider_fetch_bars(self, mock_client: MagicMock) -> None:
        # Mock the Alpaca client response
        mock_instance = mock_client.return_value
        mock_bars = MagicMock()
        # Alpaca SDK returns a dataframe via .df property
        mock_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": [datetime(2025, 1, 1)],
                "open": [150.0],
                "high": [155.0],
                "low": [149.0],
                "close": [152.0],
                "volume": [1000000],
            }
        ).set_index(["symbol", "timestamp"])
        mock_bars.df = mock_df
        mock_instance.get_stock_bars.return_value = mock_bars

        provider = AlpacaDataProvider(self.api_key, self.api_secret)
        df = provider.fetch_bars(
            ["AAPL"],
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
            timeframe=Timeframe.DAY,
        )

        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]["ticker"], "AAPL")
        self.assertEqual(df.iloc[0]["close"], 152.0)

    @patch("src.gateways.alpaca.TradingClient")
    def test_execution_backend_submit_order(
        self, mock_trading_client: MagicMock
    ) -> None:
        mock_instance = mock_trading_client.return_value
        backend = AlpacaExecutionBackend(self.api_key, self.api_secret)

        # Test successful order
        success = backend.submit_order("AAPL", 10, "BUY")
        self.assertTrue(success)
        mock_instance.submit_order.assert_called_once()

        # Test failed order
        mock_instance.submit_order.side_effect = Exception("API Error")
        success = backend.submit_order("AAPL", 10, "SELL")
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()
