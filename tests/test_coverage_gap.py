import unittest
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from src.alpha_library.features import residual_vol_20, returns_residual
from src.alpha_library.models import ReversionModel
from src.backtesting.analytics import PerformanceAnalyzer
from src.core.alpha_engine import (
    AlphaEngine,
    AlphaModel,
    ModelRunConfig,
    SignalCombiner,
    SignalProcessor,
)
from src.core.data_platform import DataPlatform
from src.core.execution_handler import (
    ExecutionHandler,
    FIXEngine,
    OrderState,
    TCAEngine,
)
from src.core.portfolio_manager import PortfolioManager
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
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)
        self.iid = self.data.get_internal_id("AAPL")
        self.ts = datetime(2025, 1, 1, 12, 0)
        self.pm = PortfolioManager(risk_aversion=1.0, max_pos=0.2)

    def test_reversion_model(self) -> None:
        model = ReversionModel()
        df = pd.DataFrame(
            [
                {
                    "timestamp": self.ts,
                    "internal_id": self.iid,
                    "returns_residual_30min": 0.05,
                    "residual_vol_20_30min": 0.02,
                }
            ]
        ).set_index("internal_id")
        signals = model.compute_signals(df)
        self.assertIn(self.iid, signals)
        self.assertAlmostEqual(signals[self.iid], -2.5)

    def test_analytics_edge_cases(self) -> None:
        # factor_attribution
        weights = {1000: 0.5, 1001: 0.5}
        returns = {1000: 0.01, 1001: -0.01}
        loadings = np.array([[1.0, 0.5], [0.5, 1.0]])
        attr = PerformanceAnalyzer.factor_attribution(
            weights, returns, loadings
        )
        self.assertIn("total", attr)
        self.assertIn("factor", attr)
        self.assertIn("selection", attr)

        # calculate_drawdown empty
        dd = PerformanceAnalyzer.calculate_drawdown(pd.Series(dtype=float))
        self.assertEqual(dd["max_dd"], 0.0)

    def test_execution_rejection(self) -> None:
        backend = MockExecutionBackend()
        handler = ExecutionHandler(backend)
        handler.vwap_execute("REJECT", 100, "BUY", slices=1)
        self.assertEqual(handler.orders[0].state, OrderState.REJECTED)

    def test_tca_slippage(self) -> None:
        # arrival_price = 0
        self.assertEqual(TCAEngine.calculate_slippage(0, 100, "BUY"), 0.0)
        # SELL sign
        slippage = TCAEngine.calculate_slippage(100, 101, "SELL")
        self.assertEqual(slippage, -100.0)

    def test_fix_engine(self) -> None:
        fix = FIXEngine("TARGET")
        self.assertTrue(fix.logon())
        self.assertTrue(fix.connected)
        order_id = fix.send_order("AAPL", 100, "BUY")
        self.assertIn("AAPL", order_id)

    def test_dataplatform_extra_coverage(self) -> None:
        # get_securities
        self.data.register_security("AAPL")
        secs = self.data.get_securities(["AAPL"])
        self.assertEqual(len(secs), 1)
        self.assertEqual(secs[0].ticker, "AAPL")

        # _update_timeseries empty
        self.data._update_timeseries("bars", pd.DataFrame())

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
                self, tickers: List[str], start: datetime, end: datetime
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
                            "ratio": 1.0,
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

        dp_prov = DataPlatform(MockProv(), clear=True)
        # First sync fills ca_df
        dp_prov.sync_data(["AAPL"], self.ts, self.ts)
        self.assertEqual(len(dp_prov.ca_df), 1)
        # Second sync with same data tests non-empty ca_df branch
        dp_prov.sync_data(["AAPL"], self.ts, self.ts)

    def test_feature_fallbacks(self) -> None:
        # Missing column
        res = returns_residual(pd.DataFrame(), "1D")
        self.assertTrue(res.empty)

        # Min samples fallback
        df = pd.DataFrame(
            [
                {
                    "timestamp": self.ts,
                    "internal_id": self.iid,
                    "returns_raw_1D": 0.01,
                    "close_1D": 100,
                }
            ]
        )
        res = returns_residual(df, "1D")
        self.assertEqual(len(res), 1)

        # residual_vol_20
        df_vol = pd.DataFrame(
            [{"internal_id": self.iid, "returns_residual_1D": 0.01}] * 20
        )
        vol = residual_vol_20(df_vol, "1D")
        self.assertEqual(len(vol), 20)

    def test_portfolio_manager_edge_cases(self) -> None:
        # check_safety branch msg_count > max_msgs
        self.pm.set_safety_limits(0, -0.1)
        self.assertFalse(self.pm.check_safety(1.0))

        # optimize empty branch
        weights = self.pm.optimize({})
        self.assertEqual(weights, {})

        # optimize returns_history is None fallback (line 91)
        self.pm.sigma = None
        weights2 = self.pm.optimize({1000: 0.1}, returns_history=None)
        self.assertEqual(weights2, {})

    def test_alpha_engine_edge_cases(self) -> None:
        # SignalProcessor.zscore std == 0
        signals = {1: 10.0, 2: 10.0}
        z = SignalProcessor.zscore(signals)
        self.assertEqual(z[1], 0.0)

        # SignalCombiner empty
        self.assertEqual(SignalCombiner.combine([]), {})
        # SignalCombiner weights is None
        combined = SignalCombiner.combine([{1: 1.0}, {1: 2.0}], weights=None)
        self.assertEqual(combined[1], 1.5)

        # AlphaModel.get_events outside context
        with self.assertRaises(RuntimeError):
            AlphaModel.get_events([self.iid])

        # AlphaEngine.run_model empty df
        class MockModel(AlphaModel):
            def compute_signals(
                self, latest: pd.DataFrame
            ) -> Dict[int, float]:
                return {}

        res = AlphaEngine.run_model(
            self.data, MockModel(), [9999], ModelRunConfig(self.ts)
        )
        self.assertEqual(res, {})

        # AlphaEngine _hydrate_features unknown feature and seen feature
        df = pd.DataFrame({"timestamp": [self.ts], "internal_id": [self.iid]})
        AlphaEngine._hydrate_features(df, ["UNKNOWN_FEATURE"])
        # seen feature
        AlphaEngine._hydrate_features(
            df, ["returns_raw_1D"], seen={"returns_raw_1D"}
        )

    def test_analytics_extra(self) -> None:
        df_empty = pd.DataFrame(columns=["timestamp", "gross_ret", "net_ret"])
        res = PerformanceAnalyzer.generate_performance_table(df_empty)
        self.assertTrue(res.empty)

    def test_alpaca_get_prices(self) -> None:
        backend = AlpacaExecutionBackend("key", "secret")
        self.assertEqual(backend.get_prices(["AAPL"]), {})


if __name__ == "__main__":
    unittest.main()
