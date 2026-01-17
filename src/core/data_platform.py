import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from arcticdb import Arctic, QueryBuilder

from src.gateways.base import DataProvider

_ARCTIC_CACHE: Dict[str, Arctic] = {}


@dataclass
class Security:
    internal_id: int
    ticker: str
    start: datetime
    end: datetime
    extra: Any


@dataclass
class Bar:
    internal_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1D"
    timestamp_knowledge: datetime = field(default_factory=datetime.now)


@dataclass
class Event:
    internal_id: int
    timestamp: datetime
    event_type: str
    value: Any
    timestamp_knowledge: datetime = field(default_factory=datetime.now)


@dataclass
class CorporateAction:
    internal_id: int
    ex_date: datetime
    type: str  # SPLIT, DIVIDEND
    value: float


@dataclass
class QueryConfig:
    start: datetime
    end: datetime
    timeframe: str = "1D"
    as_of: Optional[datetime] = None
    adjust: bool = True


class DataPlatform:
    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        db_path: str = "./.arctic_db",
        clear: bool = False,
        aggregate_to: Optional[List[str]] = None,
    ) -> None:
        self.provider = provider
        self.aggregate_to = aggregate_to
        self.arctic = _ARCTIC_CACHE.setdefault(
            db_path, Arctic(f"lmdb://{db_path}")
        )
        if clear and "platform" in self.arctic.list_libraries():
            self.arctic.delete_library("platform")
        self.lib = self.arctic.get_library("platform", create_if_missing=True)
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Initializes metadata tables."""
        self.meta: Dict[str, pd.DataFrame] = {}
        defaults = {
            "sec_df": (
                Security,
                {"internal_id": "int64", "start": "datetime64[ns]"},
            ),
            "ca_df": (
                CorporateAction,
                {
                    "internal_id": "int64",
                    "ex_date": "datetime64[ns]",
                    "value": "float64",
                },
            ),
        }

        for name, (cls, dtypes) in defaults.items():
            if not self.lib.has_symbol(name):
                self.lib.write(
                    name,
                    pd.DataFrame(columns=[f.name for f in fields(cls)]).astype(
                        dtypes
                    ),
                )
            self.meta[name] = self.lib.read(name).data

        # Decode JSON in memory
        if not self.meta["sec_df"].empty:
            self.meta["sec_df"]["extra"] = self.meta["sec_df"]["extra"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        self._id_counter = (
            int(self.meta["sec_df"].internal_id.max() + 1)
            if not self.meta["sec_df"].empty
            else 1000
        )

    def _save_meta(self, name: str) -> None:
        df = self.meta[name].copy()
        if name == "sec_df" and not df.empty:
            df["extra"] = df["extra"].apply(json.dumps)
        self.lib.write(name, df)

    def _write_ts(self, symbol: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        current_data = (
            self.lib.read(symbol).data
            if self.lib.has_symbol(symbol)
            else pd.DataFrame()
        )
        if current_data.empty:
            updated_data = df.sort_index()
        else:
            updated_data = pd.concat([current_data, df]).sort_index()
        self.lib.write(symbol, updated_data)

    def _update_timeseries(self, symbol: str, df: pd.DataFrame) -> None:
        """Alias for backward compatibility in tests."""
        self._write_ts(symbol, df)

    # --- Security Master ---

    @property
    def sec_df(self) -> pd.DataFrame:
        return self.meta["sec_df"]

    @sec_df.setter
    def sec_df(self, value: pd.DataFrame) -> None:
        self.meta["sec_df"] = value

    @property
    def ca_df(self) -> pd.DataFrame:
        return self.meta["ca_df"]

    @ca_df.setter
    def ca_df(self, value: pd.DataFrame) -> None:
        self.meta["ca_df"] = value

    def register_security(
        self,
        ticker: str,
        internal_id: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        **kwargs: Any,
    ) -> int:
        start_date, end_date = (
            start or datetime(1900, 1, 1),
            end or datetime(2200, 1, 1),
        )
        df = self.meta["sec_df"]
        existing_security = df[
            (df.ticker == ticker)
            & (df.start <= end_date)
            & (df.end >= start_date)
        ]

        if internal_id is None and not existing_security.empty:
            return int(existing_security.iloc[0].internal_id)

        iid = internal_id or self._id_counter
        if internal_id is None:
            self._id_counter += 1

        new_row = pd.DataFrame(
            [asdict(Security(iid, ticker, start_date, end_date, kwargs))]
        )

        if df.empty:
            self.meta["sec_df"] = new_row
        else:
            self.meta["sec_df"] = pd.concat([df, new_row], ignore_index=True)

        self._save_meta("sec_df")
        return iid

    def get_internal_id(
        self, ticker: str, date: Optional[datetime] = None
    ) -> int:
        return self.register_security(ticker, start=date or datetime.now())

    def get_securities(
        self, tickers: Optional[List[str]] = None
    ) -> List[Security]:
        df = self.meta["sec_df"]
        records = (df[df.ticker.isin(tickers)] if tickers else df).to_dict(
            "records"
        )
        return [Security(**r) for r in records]

    def get_universe(self, date: datetime) -> List[int]:
        df = self.meta["sec_df"]
        uids = (
            df[(df.start <= date) & (df.end >= date)]
            .internal_id.unique()
            .tolist()
        )
        return [int(x) for x in uids]

    @property
    def reverse_ism(self) -> Dict[int, str]:
        return (
            cast(
                Dict[int, str],
                self.meta["sec_df"].set_index("internal_id").ticker.to_dict(),
            )
            if not self.meta["sec_df"].empty
            else {}
        )

    # --- Data IO ---

    def add_bars(self, bars: List[Bar]) -> None:
        """
        Add bars to the platform.
        Statelessly forms higher timeframe bars if self.aggregate_to is set.
        """
        if not bars:
            return

        # 1. Persist the bars provided
        df_new = pd.DataFrame([asdict(b) for b in bars]).set_index("timestamp")
        self._write_ts("bars", df_new)

        if not self.aggregate_to:
            return

        # 2. Aggregation Logic (Stateless)
        for bar in bars:
            for target_tf in self.aggregate_to:
                if target_tf == bar.timeframe:
                    continue

                # Determine the start of the current aggregation window
                freq = target_tf.replace("min", "min")
                window_start = (
                    pd.Timestamp(bar.timestamp).floor(freq).to_pydatetime()
                )

                # Query DB for all raw bars in this window
                query = QueryBuilder()
                query = query[
                    (query.internal_id == bar.internal_id)
                    & (query.timeframe == bar.timeframe)
                    & (query.timestamp >= window_start)
                    & (query.timestamp <= bar.timestamp)
                ]

                window_df = self.lib.read("bars", query_builder=query).data

                # Check if we have enough bars to form the target
                try:
                    target_mins = int(target_tf.replace("min", ""))
                    source_mins = int(bar.timeframe.replace("min", ""))
                    required_count = target_mins // source_mins

                    if len(window_df) >= required_count:
                        agg_bar = Bar(
                            internal_id=bar.internal_id,
                            timestamp=window_start,
                            open=float(window_df.open.iloc[0]),
                            high=float(window_df.high.max()),
                            low=float(window_df.low.min()),
                            close=float(window_df.close.iloc[-1]),
                            volume=float(window_df.volume.sum()),
                            timeframe=target_tf,
                            timestamp_knowledge=bar.timestamp_knowledge,
                        )
                        self._write_ts(
                            "bars",
                            pd.DataFrame([asdict(agg_bar)]).set_index(
                                "timestamp"
                            ),
                        )
                except (ValueError, ZeroDivisionError, IndexError):
                    continue

    def add_events(self, events: List[Event]) -> None:
        df = pd.DataFrame([asdict(e) for e in events])
        if not df.empty:
            df["value"] = df["value"].apply(json.dumps)
        self._write_ts("events", df.set_index("timestamp"))

    def add_ca(self, ca: CorporateAction) -> None:
        new_row = pd.DataFrame([asdict(ca)])
        if self.meta["ca_df"].empty:
            self.meta["ca_df"] = new_row
        else:
            self.meta["ca_df"] = pd.concat(
                [self.meta["ca_df"], new_row]
            ).drop_duplicates()
        self._save_meta("ca_df")

    def get_bars(self, iids: List[int], cfg: QueryConfig) -> pd.DataFrame:
        if not self.lib.has_symbol("bars"):
            return pd.DataFrame()

        query = QueryBuilder()
        query = query[
            query.internal_id.isin(iids)
            & (query.timeframe == cfg.timeframe)
            & (query.timestamp >= cfg.start)
            & (query.timestamp <= cfg.end)
            & (query.timestamp_knowledge <= (cfg.as_of or datetime.now()))
        ]

        df = self.lib.read("bars", query_builder=query).data.reset_index()
        if df.empty:
            return df

        # PIT: Deduplicate by taking the latest knowledge time
        df = (
            df.sort_values("timestamp_knowledge")
            .groupby(["internal_id", "timestamp"])
            .last()
            .reset_index()
        )

        # Cast to float64 to avoid warnings on adjustments (e.g. Difference)
        cols = ["open", "high", "low", "close"]
        df[cols] = df[cols].astype("float64")

        if cfg.adjust and not self.meta["ca_df"].empty:
            corporate_actions = self.meta["ca_df"]
            corporate_actions = corporate_actions[
                corporate_actions.internal_id.isin(iids)
                & (corporate_actions.ex_date <= cfg.end)
            ].sort_values("ex_date", ascending=False)

            for iid in iids:
                # Track cumulative ratio adjustment to scale difference adj
                # (e.g., div before 2:1 split should be adjusted by 0.5)
                adj_factor = 1.0

                for _, r in corporate_actions[
                    corporate_actions.internal_id == iid
                ].iterrows():
                    mask = (df.internal_id == iid) & (df.timestamp < r.ex_date)
                    cols = ["open", "high", "low", "close"]

                    if r.type == "SPLIT":
                        ratio = 1.0 / r.value
                        df.loc[mask, cols] *= ratio
                        adj_factor *= ratio
                    elif r.type == "DIVIDEND":
                        # Difference adjustment scaled by future ratios
                        df.loc[mask, cols] -= r.value * adj_factor

        return df.rename(
            columns={
                c: f"{c}_{cfg.timeframe}"
                for c in ["open", "high", "low", "close", "volume"]
            }
        )

    def get_events(
        self,
        iids: List[int],
        types: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Event]:
        if not self.lib.has_symbol("events"):
            return []
        query = QueryBuilder()
        query = query[
            query.internal_id.isin(iids)
            & (query.timestamp_knowledge <= (as_of or datetime.now()))
        ]
        if start:
            query = query[query.timestamp >= start]
        if end:
            query = query[query.timestamp <= end]
        if types:
            query = query[query.event_type.isin(types)]

        df = self.lib.read("events", query_builder=query).data.reset_index()
        return [
            Event(**{**r, "value": json.loads(r["value"])})
            for r in df.to_dict("records")
        ]

    def sync_data(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> None:
        if not self.provider:
            return

        def with_ids(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
            df["internal_id"] = df.apply(
                lambda r: self.get_internal_id(r.ticker, r[date_col]), axis=1
            )
            return df.drop(columns=["ticker"])

        # Bars
        bars = self.provider.fetch_bars(tickers, start, end)
        if not bars.empty:
            timeframe = bars.get("timeframe", pd.Series(["1D"] * len(bars)))
            df = with_ids(bars, "timestamp").assign(
                timestamp_knowledge=datetime.now(), timeframe=timeframe
            )
            self._write_ts("bars", df.set_index("timestamp"))

        # Corporate Actions
        corporate_actions = self.provider.fetch_corporate_actions(
            tickers, start, end
        )
        if not corporate_actions.empty:
            # Map amount/ratio to 'value' depending on type
            if "value" not in corporate_actions.columns:
                # Fallback mapping if provider uses ratio/amount
                corporate_actions["value"] = corporate_actions.apply(
                    lambda r: r.get("ratio") or r.get("amount") or 0.0, axis=1
                )

            new_corporate_actions = with_ids(corporate_actions, "ex_date")
            new_ca_df = new_corporate_actions[
                ["internal_id", "ex_date", "type", "value"]
            ].drop_duplicates()

            if self.meta["ca_df"].empty:
                self.meta["ca_df"] = new_ca_df
            else:
                self.meta["ca_df"] = pd.concat(
                    [self.meta["ca_df"], new_ca_df]
                ).drop_duplicates()
            self._save_meta("ca_df")

        # Events
        events_df = self.provider.fetch_events(tickers, start, end)
        if not events_df.empty:
            df = with_ids(events_df, "timestamp").assign(
                timestamp_knowledge=datetime.now()
            )
            df["value"] = df["value"].apply(json.dumps)
            self._write_ts("events", df.set_index("timestamp"))
