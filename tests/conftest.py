from datetime import datetime, timedelta
from typing import Any, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.core.data_platform import DataPlatform
from src.core.types import Bar, Timeframe
from src.gateways.base import (
    BarProvider,
    CorporateActionProvider,
    EventProvider,
    ExecutionBackend,
)


@pytest.fixture
def arctic_db_path(tmp_path: Any) -> str:
    """Provides a temporary directory for ArcticDB."""
    return str(tmp_path / "arctic_db")


@pytest.fixture
def data_platform(arctic_db_path: str) -> DataPlatform:
    """Provides a clean DataPlatform instance."""
    return DataPlatform(db_path=arctic_db_path, clear=True)


@pytest.fixture
def populated_platform(data_platform: DataPlatform) -> Any:
    """Provides a DataPlatform with 2 registered securities and bars."""
    ts = datetime(2025, 1, 1, 12, 0)
    iids = [data_platform.register_security(t) for t in ["AAPL", "MSFT"]]

    for iid in iids:
        bars = []
        for i in range(30):
            bar = Bar(
                iid,
                ts - timedelta(minutes=30 * i),
                100.0 + i,
                101.0 + i,
                99.0 + i,
                100.0 + i,
                1000,
                timeframe=Timeframe.MIN_30,
            )
            bars.append(bar)
        data_platform.add_bars(bars)
    return data_platform, iids[0], ts


class MockProvider(BarProvider, CorporateActionProvider, EventProvider):
    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = Timeframe.DAY,
    ) -> pd.DataFrame:
        dates = pd.date_range(start, end, freq=timeframe.pandas_freq)
        data = []
        for t in tickers:
            for i, dt in enumerate(dates):
                data.append(
                    {
                        "ticker": t,
                        "timestamp": dt,
                        "open": 100.0 + i,
                        "high": 101.0 + i,
                        "low": 99.0 + i,
                        "close": 100.0 + i,
                        "volume": 1000,
                        "timeframe": timeframe.value,
                    }
                )
        return pd.DataFrame(data)

    def fetch_corporate_actions(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": t,
                    "ex_date": start,
                    "type": "DIVIDEND",
                    "value": 1.0,
                }
                for t in tickers
            ]
        )

    def fetch_events(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": t,
                    "timestamp": start,
                    "event_type": "TEST_EVENT",
                    "value": {"data": 1},
                }
                for t in tickers
            ]
        )


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def mock_backend() -> Any:
    backend = MagicMock(spec=ExecutionBackend)
    backend.submit_order.return_value = True
    backend.get_positions.return_value = {}
    backend.get_prices.return_value = {}
    return backend
