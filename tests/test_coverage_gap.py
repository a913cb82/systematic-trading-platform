import time
import unittest
from datetime import datetime
from typing import Dict, List
from unittest.mock import patch

import pandas as pd

from src.alpha_library.features import residual_vol_20, returns_residual
from src.core.alpha_engine import (
    AlphaEngine,
    AlphaModel,
    ModelRunConfig,
    SignalCombiner,
    SignalProcessor,
)
from src.core.data_platform import DataPlatform, Event
from src.core.execution_handler import (
    ExecutionHandler,
    FIXEngine,
    Order,
    OrderState,
    TCAEngine,
)
from src.core.portfolio_manager import PortfolioManager
from src.core.types import Timeframe
from src.gateways.alpaca import AlpacaExecutionBackend
from src.gateways.base import DataProvider, ExecutionBackend


class MockExecutionBackend(ExecutionBackend):
    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        if ticker == "REJECT":
            return False
        return True

    def get_positions(self) -> Dict[str, float]:
        return {}

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: 100.0 for t in tickers}


class TestCoverageGap(unittest.TestCase):
    """Fills small holes in coverage found by reports."""

    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)
        self.ts = datetime(2025, 1, 1, 12, 0)

    def test_alpha_engine_edge_cases(self) -> None:
        # Empty signals zscore
        self.assertEqual(SignalProcessor.zscore({}), {})

        # Zero std zscore
        signals = {1: 10.0, 2: 10.0}
        res = SignalProcessor.zscore(signals)
        self.assertEqual(res, {1: 0.0, 2: 0.0})

        # Combiner empty list
        self.assertEqual(SignalCombiner.combine([]), {})

        # AlphaEngine run_model empty bars
        class EmptyModel(AlphaModel):
            def compute_signals(
                self, latest: pd.DataFrame
            ) -> Dict[int, float]:
                return {}

        res = AlphaEngine.run_model(
            self.data,
            EmptyModel(),
            [999],
            ModelRunConfig(self.ts, timeframe=Timeframe.DAY),
        )
        self.assertEqual(res, {})

    def test_execution_state_machine(self) -> None:
        # Partial fill state transition
        o = Order("AAPL", 100, "BUY")
        o.update(50)
        self.assertEqual(o.state, OrderState.PARTIAL)
        o.update(50)
        self.assertEqual(o.state, OrderState.FILLED)

    def test_execution_rejection(self) -> None:
        backend = MockExecutionBackend()
        handler = ExecutionHandler(backend)
        handler.vwap_execute("REJECT", 100, "BUY", slices=1, interval=0)

        # Wait for worker
        max_wait = 1.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if handler.orders[0].state == OrderState.REJECTED:
                break
            time.sleep(0.01)

        self.assertEqual(handler.orders[0].state, OrderState.REJECTED)

    def test_tca_arrival_zero(self) -> None:
        self.assertEqual(TCAEngine.calculate_slippage(0, 100, "BUY"), 0.0)

    def test_fix_engine_stubs(self) -> None:
        fix = FIXEngine("TEST")
        self.assertTrue(fix.logon())
        self.assertIn("AAPL", fix.send_order("AAPL", 10, "BUY"))

    def test_dataplatform_extra_coverage(self) -> None:
        # get_securities
        self.data.register_security("AAPL")
        secs = self.data.get_securities(["AAPL"])
        self.assertEqual(len(secs), 1)
        self.assertEqual(secs[0].ticker, "AAPL")

        # _write empty
        self.data._write("bars", pd.DataFrame())

        # get_events empty and with end date
        res = self.data.get_events([9999], end=self.ts)
        self.assertEqual(res, [])

        # sync_data coverage
        # no provider
        dp_no_prov = DataPlatform()
        dp_no_prov.sync_data(["AAPL"], self.ts, self.ts)

        # with provider, test ca_df empty branch and persist_metadata at end
        class MockProv(DataProvider):
            def fetch_bars(
                self,
                tickers: List[str],
                start: datetime,
                end: datetime,
                timeframe: Timeframe = Timeframe.DAY,
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "ticker": "AAPL",
                            "timestamp": start,
                            "open": 100,
                            "high": 101,
                            "low": 99,
                            "close": 100,
                            "volume": 1000,
                            "timeframe": timeframe,
                        }
                    ]
                )

            def fetch_corporate_actions(
                self, tickers: List[str], start: datetime, end: datetime
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "ticker": "AAPL",
                            "ex_date": start,
                            "type": "DIV",
                            "value": 1.0,
                        }
                    ]
                )

            def fetch_events(
                self, tickers: List[str], start: datetime, end: datetime
            ) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "ticker": "AAPL",
                            "timestamp": start,
                            "event_type": "E",
                            "value": {},
                        }
                    ]
                )

        dp_prov = DataPlatform(
            MockProv(), db_path="./.arctic_test_cov", clear=True
        )
        # First sync fills ca_df
        dp_prov.sync_data(["AAPL"], self.ts, self.ts)
        # Second sync tests concat branch
        dp_prov.sync_data(["AAPL"], self.ts, self.ts)

        # add_events coverage
        self.data.add_events([Event(1000, self.ts, "TYPE", {"key": "val"})])

        # add_bars empty branch
        self.data.add_bars([])

    def test_features_edge_cases(self) -> None:
        # returns_residual empty df
        res = returns_residual(pd.DataFrame(), Timeframe.MIN_30)
        self.assertTrue(res.empty)

        # returns_residual small universe (skip branch)
        df = pd.DataFrame(
            {
                "timestamp": [self.ts] * 2,
                "internal_id": [1, 2],
                "returns_raw_30min": [0.01, 0.02],
            }
        )
        res = returns_residual(df, Timeframe.MIN_30)
        self.assertTrue(res.isna().all())

        # residual_vol_20 coverage
        df_vol = pd.DataFrame(
            {
                "internal_id": [1] * 5,
                "returns_residual_30min": [0.1, 0.2, 0.1, 0.2, 0.1],
            }
        )
        res_vol = residual_vol_20(df_vol, Timeframe.MIN_30)
        self.assertEqual(len(res_vol), 5)

    def test_portfolio_manager_edge_cases(self) -> None:
        # optimize empty forecasts
        weights = self.pm.optimize({})
        self.assertEqual(weights, {})

        # optimize with missing returns_history when needed
        self.pm.sigma = None
        weights2 = self.pm.optimize({1000: 0.1}, returns_history=None)
        self.assertEqual(weights2, {})

    def test_alpaca_plugin_extra(self) -> None:
        # Mock API errors in get_prices and get_positions
        with patch("src.gateways.alpaca.TradingClient") as mock_trade:
            inst = mock_trade.return_value
            inst.get_all_positions.side_effect = Exception("err")
            backend = AlpacaExecutionBackend("k", "s")
            self.assertEqual(backend.get_positions(), {})

        with patch(
            "src.gateways.alpaca.StockHistoricalDataClient"
        ) as mock_hist:
            inst = mock_hist.return_value
            inst.get_stock_latest_quote.side_effect = Exception("err")
            backend = AlpacaExecutionBackend("k", "s")
            self.assertEqual(backend.get_prices(["AAPL"]), {})

    @property
    def pm(self) -> PortfolioManager:
        return PortfolioManager()


if __name__ == "__main__":
    unittest.main()
