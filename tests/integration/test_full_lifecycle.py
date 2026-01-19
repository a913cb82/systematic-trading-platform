import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core.data_platform import DataPlatform
from src.core.execution_handler import ExecutionHandler, OrderState
from src.core.portfolio_manager import PortfolioManager
from src.gateways.base import (
    BarProvider,
    CorporateActionProvider,
    EventProvider,
    ExecutionBackend,
)

# Constants to avoid magic values
AAPL_TICKER = "AAPL"
START_PRICE = 100.0
CAPITAL = 1000000.0


class MockPlugin(
    BarProvider, CorporateActionProvider, EventProvider, ExecutionBackend
):
    """Unified mock provider for integration testing."""

    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: Any = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_corporate_actions(
        self, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_events(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()

    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        return True

    def get_positions(self) -> Dict[str, float]:
        return {}

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: START_PRICE for t in tickers}


def test_system_integrates_alpha_portfolio_and_execution_for_full_rebalance(
    arctic_db_path: str, mock_provider: Any
) -> None:
    plugin = MockPlugin()
    data = DataPlatform(plugin, db_path=arctic_db_path, clear=True)
    pm = PortfolioManager(tc_penalty=0.01)
    exec_h = ExecutionHandler(backend=plugin)

    tickers = ["AAPL", "MSFT", "GOOG"]
    iids = [data.get_internal_id(t) for t in tickers]
    reverse_ism = data.reverse_ism

    data.sync_data(tickers, datetime(2025, 1, 1), datetime(2025, 1, 1))
    weights = pm.optimize(
        {iid: 0.5 for iid in iids}, np.random.randn(20, 3) * 0.01
    )

    prices = plugin.get_prices(tickers)
    goal_positions = {}
    for iid, weight in weights.items():
        ticker = reverse_ism[iid]
        goal_positions[ticker] = (weight * CAPITAL) / prices[ticker]

    exec_h.rebalance(goal_positions, interval=0)
    max_wait, start_time = 1.0, time.time()
    while time.time() - start_time < max_wait:
        if all(o.state == OrderState.FILLED for o in exec_h.orders):
            break
        time.sleep(0.01)
    assert len(exec_h.orders) > 0


def test_portfolio_execution_interface_compatibility(
    mock_backend: Any,
) -> None:
    pm = PortfolioManager()
    weights = pm.optimize({1000: 0.05}, np.random.randn(20, 1) * 0.01)
    handler = ExecutionHandler(mock_backend)
    handler.rebalance({"AAPL": (weights[1000] * 100000) / 150.0}, interval=0)
