import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.alpaca_plugin import AlpacaDataProvider, AlpacaExecutionBackend


class TestAlpacaPlugin(unittest.TestCase):
    def test_alpaca_data_provider_mock(self) -> None:
        with patch(
            "src.alpaca_plugin.StockHistoricalDataClient"
        ) as mock_client:
            provider = AlpacaDataProvider("key", "secret")
            mock_bars = MagicMock()
            mock_bars.df = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [("AAPL", datetime(2025, 1, 1))],
                    names=["symbol", "timestamp"],
                ),
                data={
                    "open": [100],
                    "high": [101],
                    "low": [99],
                    "close": [100.5],
                    "volume": [1000],
                },
            )
            mock_client.return_value.get_stock_bars.return_value = mock_bars

            df = provider.fetch_bars(
                ["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 2)
            )
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["ticker"], "AAPL")

            ca = provider.fetch_corporate_actions(
                ["AAPL"], datetime(2025, 1, 1), datetime(2025, 1, 2)
            )
            self.assertTrue(ca.empty)

    def test_alpaca_execution_backend_mock(self) -> None:
        with patch("src.alpaca_plugin.TradingClient") as mock_client:
            backend = AlpacaExecutionBackend("key", "secret")

            # Submit Order
            mock_client.return_value.submit_order.return_value = True
            res = backend.submit_order("AAPL", 10, "BUY")
            self.assertTrue(res)

            # Submit Order Error
            mock_client.return_value.submit_order.side_effect = Exception(
                "error"
            )
            res = backend.submit_order("AAPL", 10, "SELL")
            self.assertFalse(res)

            # Get Positions
            mock_pos = MagicMock()
            mock_pos.symbol = "AAPL"
            mock_pos.qty = "15"
            mock_client.return_value.get_all_positions.return_value = [
                mock_pos
            ]
            pos = backend.get_positions()
            self.assertEqual(pos["AAPL"], 15.0)

            # Get Prices
            self.assertEqual(backend.get_prices(["AAPL"]), {})


if __name__ == "__main__":
    unittest.main()
