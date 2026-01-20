import warnings
from datetime import datetime, timedelta
from typing import Any, List

import pandas as pd

from src.core.data_platform import CorporateAction, DataPlatform, Event
from src.core.types import Bar, QueryConfig, Timeframe
from src.gateways.base import BarProvider, CorporateActionProvider

# Constants to avoid magic values
AAPL_TICKER = "AAPL"
MSFT_TICKER = "MSFT"
FB_TICKER = "FB"
META_TICKER = "META"
TECH_SECTOR = "Tech"
DEFAULT_VOLUME = 1000000
SPLIT_RATIO = 2.0
DIVIDEND_VALUE = 1.0
EARNINGS_EPS = 1.5
TEST_IID = 1000
PERSISTED_IID = 5000
HIGH_PRICE = 152.0
ADJUSTED_PRICE = 50.0
RESTATED_PRICE = 105.0
AGGREGATED_HIGH = 130.0
AGGREGATED_VOLUME = 3000.0
BASE_PRICE = 100.0
U_SIZE_2 = 2
BAR_HIST_2 = 2


def test_dataplatform_persists_security_metadata(
    data_platform: DataPlatform,
) -> None:
    iid = data_platform.register_security(AAPL_TICKER, sector=TECH_SECTOR)
    assert data_platform.get_internal_id(AAPL_TICKER) == iid
    secs = data_platform.get_securities([AAPL_TICKER])
    assert len(secs) == 1
    assert secs[0].extra["sector"] == TECH_SECTOR


def test_dataplatform_stores_and_retrieves_bars(
    data_platform: DataPlatform,
) -> None:
    ts = datetime(2025, 1, 1, 12, 0)
    iid = data_platform.register_security(AAPL_TICKER)
    bar = Bar(
        iid,
        ts,
        150.0,
        155.0,
        149.0,
        HIGH_PRICE,
        DEFAULT_VOLUME,
        timeframe=Timeframe.DAY,
    )
    data_platform.add_bars([bar])
    query = QueryConfig(start=ts, end=ts, timeframe=Timeframe.DAY)
    df = data_platform.get_bars([iid], query)
    assert not df.empty
    assert df.iloc[0]["close_1D"] == HIGH_PRICE


def test_dataplatform_adjusts_prices_for_splits(
    data_platform: DataPlatform,
) -> None:
    ts1, ts2 = datetime(2025, 1, 1), datetime(2025, 1, 2)
    iid = data_platform.register_security(AAPL_TICKER)
    bar = Bar(iid, ts1, 100, 100, 100, 100, 1000, timeframe=Timeframe.DAY)
    data_platform.add_bars([bar])
    data_platform.add_ca(CorporateAction(iid, ts2, "SPLIT", SPLIT_RATIO))
    query = QueryConfig(
        start=ts1, end=ts2, timeframe=Timeframe.DAY, adjust=True
    )
    df = data_platform.get_bars([iid], query)
    assert df[df.timestamp == ts1].iloc[0]["close_1D"] == ADJUSTED_PRICE


def test_dataplatform_stores_and_retrieves_events(
    data_platform: DataPlatform,
) -> None:
    ts = datetime(2025, 1, 1)
    iid = data_platform.register_security(AAPL_TICKER)
    ev = Event(iid, ts, "EARNINGS", {"eps": EARNINGS_EPS})
    data_platform.add_events([ev])
    events = data_platform.get_events([iid], types=["EARNINGS"])
    assert len(events) == 1
    assert events[0].value["eps"] == EARNINGS_EPS


def test_dataplatform_restores_state_from_db(arctic_db_path: str) -> None:
    ts, iid = datetime(2025, 1, 1, 12, 0), TEST_IID
    dp1 = DataPlatform(db_path=arctic_db_path, clear=True)
    bar = Bar(iid, ts, 100, 101, 99, 100, 1000, timeframe=Timeframe.DAY)
    dp1.add_bars([bar])
    dp2 = DataPlatform(db_path=arctic_db_path, clear=False)
    query = QueryConfig(start=ts, end=ts, timeframe=Timeframe.DAY)
    df = dp2.get_bars([iid], query)
    assert df.iloc[0]["close_1D"] == BASE_PRICE


def test_dataplatform_respects_bitemporal_as_of_time(
    data_platform: DataPlatform,
) -> None:
    iid = data_platform.register_security(AAPL_TICKER)
    t1 = datetime(2025, 1, 1, 12, 0)
    bar1 = Bar(iid, t1, 100, 101, 99, 100, 1000, timestamp_knowledge=t1)
    data_platform.add_bars([bar1])

    tk2 = t1 + timedelta(hours=1)
    bar2 = Bar(
        iid, t1, 100, 105, 99, RESTATED_PRICE, 1000, timestamp_knowledge=tk2
    )
    data_platform.add_bars([bar2])

    cfg = QueryConfig(start=t1, end=t1, timeframe=Timeframe.DAY, as_of=t1)
    df1 = data_platform.get_bars([iid], cfg)
    assert df1.iloc[0]["close_1D"] == BASE_PRICE
    cfg.as_of = tk2
    df2 = data_platform.get_bars([iid], cfg)
    assert df2.iloc[0]["close_1D"] == RESTATED_PRICE


def test_dataplatform_aggregates_intraday_bars(
    data_platform: DataPlatform,
) -> None:
    iid = data_platform.register_security(AAPL_TICKER)
    ts = datetime(2025, 1, 1, 12, 0)
    bars = []
    for i in range(30):
        bar = Bar(
            iid,
            ts + timedelta(minutes=i),
            100 + i,
            100 + i + 1,
            100 + i - 1,
            100 + i,
            100,
            timeframe=Timeframe.MINUTE,
        )
        bars.append(bar)
    data_platform.add_bars(bars)

    window_start = pd.Timestamp(ts).floor("30min").to_pydatetime()
    query = QueryConfig(
        start=window_start,
        end=window_start + timedelta(minutes=29),
        timeframe=Timeframe.MIN_30,
    )
    df = data_platform.get_bars([iid], query)
    assert df.iloc[0]["high_30min"] == AGGREGATED_HIGH


def test_dataplatform_reconstructs_point_in_time_universe(
    data_platform: DataPlatform,
) -> None:
    t1, t2 = datetime(2025, 1, 1), datetime(2025, 2, 1)
    data_platform.register_security(AAPL_TICKER, start=t1)
    data_platform.register_security(MSFT_TICKER, start=t2)
    u1 = data_platform.get_universe(t1 + timedelta(days=1))
    assert len(u1) == 1
    u2 = data_platform.get_universe(t2 + timedelta(days=1))
    assert len(u2) == U_SIZE_2


def test_dataplatform_syncs_data_from_multiple_providers(
    arctic_db_path: str, mock_provider: Any
) -> None:
    dp = DataPlatform(mock_provider, db_path=arctic_db_path, clear=True)
    iid = dp.register_security(AAPL_TICKER)
    dp.sync_data([AAPL_TICKER], datetime(2025, 1, 1), datetime(2025, 1, 1))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*BlockManagerUnconsolidated.*",
        )
        assert dp.lib.read("bars").data.iloc[0]["internal_id"] == iid

    assert dp.ca_df.iloc[0]["internal_id"] == iid


def test_dataplatform_ignores_empty_bar_lists(
    data_platform: DataPlatform,
) -> None:
    data_platform.add_bars([])
    assert not data_platform.lib.has_symbol("bars")


def test_dataplatform_ignores_empty_dataframe_writes(
    data_platform: DataPlatform,
) -> None:
    data_platform._write("bars", pd.DataFrame())
    assert not data_platform.lib.has_symbol("bars")


def test_dataplatform_returns_empty_list_for_missing_events(
    data_platform: DataPlatform,
) -> None:
    assert data_platform.get_events([9999], end=datetime.now()) == []


def test_dataplatform_separates_internal_ids_for_symbol_changes(
    data_platform: DataPlatform,
) -> None:
    iid = data_platform.get_internal_id(FB_TICKER)
    iid_meta = data_platform.get_internal_id(META_TICKER)
    assert iid != iid_meta


def test_dataplatform_resolves_ticker_from_bar_metadata(
    data_platform: DataPlatform,
) -> None:
    iid = data_platform.register_security(AAPL_TICKER)
    ts = datetime(2025, 1, 1, 12, 0)
    b = Bar(
        internal_id=0,
        timestamp=ts,
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1000,
        timeframe=Timeframe.MINUTE,
    )
    b._ticker = AAPL_TICKER
    data_platform.add_bars([b])
    query = QueryConfig(start=ts, end=ts, timeframe=Timeframe.MINUTE)
    df = data_platform.get_bars([iid], query)
    assert not df.empty
    assert df.iloc[0]["close_1min"] == BASE_PRICE


def test_dataplatform_preserves_bitemporal_integrity_on_write(
    data_platform: DataPlatform,
) -> None:
    iid = data_platform.register_security(AAPL_TICKER)
    ts = datetime(2025, 1, 1, 12, 0)
    tk1, tk2 = ts + timedelta(minutes=1), ts + timedelta(minutes=2)
    bar1 = Bar(
        iid,
        ts,
        100,
        101,
        99,
        100,
        1000,
        timeframe=Timeframe.MINUTE,
        timestamp_knowledge=tk1,
    )
    data_platform.add_bars([bar1])
    bar2 = Bar(
        iid,
        ts,
        100,
        105,
        99,
        RESTATED_PRICE,
        1000,
        timeframe=Timeframe.MINUTE,
        timestamp_knowledge=tk2,
    )
    data_platform.add_bars([bar2])
    query = QueryConfig(start=ts, end=ts, timeframe=Timeframe.MINUTE)
    df = data_platform.get_bars([iid], query)
    assert len(df) == 1
    assert df.iloc[0]["close_1min"] == RESTATED_PRICE

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*BlockManagerUnconsolidated.*",
        )
        raw = data_platform.lib.read("bars").data

    assert len(raw) == BAR_HIST_2


def test_dataplatform_sync_resolves_internal_ids(arctic_db_path: str) -> None:
    class MockProv(BarProvider, CorporateActionProvider):
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
                        "ticker": AAPL_TICKER,
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
                        "ticker": AAPL_TICKER,
                        "ex_date": start,
                        "type": "DIV",
                        "value": DIVIDEND_VALUE,
                    }
                ]
            )

    dp = DataPlatform(MockProv(), db_path=arctic_db_path, clear=True)
    expected_iid = dp.register_security(AAPL_TICKER)
    dp.sync_data([AAPL_TICKER], datetime(2025, 1, 1), datetime(2025, 1, 1))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*BlockManagerUnconsolidated.*",
        )
        assert dp.lib.read("bars").data.iloc[0]["internal_id"] == expected_iid

    assert dp.ca_df.iloc[0]["internal_id"] == expected_iid


def test_dataplatform_metadata_persistence_audit(arctic_db_path: str) -> None:
    dp1 = DataPlatform(db_path=arctic_db_path, clear=True)
    dp1.register_security(AAPL_TICKER, internal_id=PERSISTED_IID)
    dp2 = DataPlatform(db_path=arctic_db_path, clear=False)
    assert dp2.get_internal_id(AAPL_TICKER) == PERSISTED_IID


def test_dataplatform_aggregation_multi_chunk_audit(
    data_platform: DataPlatform,
) -> None:
    iid = data_platform.register_security(AAPL_TICKER)
    ts = datetime(2025, 1, 1, 12, 0)
    bars = []
    for i in range(30):
        bar = Bar(
            iid,
            ts + timedelta(minutes=i),
            100 + i,
            100 + i + 1,
            100 + i - 1,
            100 + i,
            100,
            timeframe=Timeframe.MINUTE,
        )
        bars.append(bar)
    data_platform.add_bars(bars[:15])
    data_platform.add_bars(bars[15:])

    window_start = pd.Timestamp(ts).floor("30min").to_pydatetime()
    query = QueryConfig(
        start=window_start,
        end=window_start + timedelta(minutes=29),
        timeframe=Timeframe.MIN_30,
    )
    df = data_platform.get_bars([iid], query)
    assert df.iloc[0]["volume_30min"] == AGGREGATED_VOLUME
