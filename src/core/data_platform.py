import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from arcticdb import Arctic, QueryBuilder

from src.core.types import (
    Bar,
    CorporateAction,
    Event,
    QueryConfig,
    Security,
    Timeframe,
)
from src.gateways.base import DataProvider

_ARCTIC_CACHE: Dict[str, Arctic] = {}


class DataPlatform:
    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        db_path: str = "./.arctic_db",
        clear: bool = False,
    ) -> None:
        self.provider = provider
        self.arctic = _ARCTIC_CACHE.setdefault(
            db_path, Arctic(f"lmdb://{db_path}")
        )
        if clear and "platform" in self.arctic.list_libraries():
            self.arctic.delete_library("platform")
        self.lib = self.arctic.get_library("platform", create_if_missing=True)
        # Simple ID counter state
        self._id_counter = 1000
        if self.lib.has_symbol("sec_df"):
            df = self.lib.read("sec_df").data
            if not df.empty:
                self._id_counter = int(df.internal_id.max() + 1)

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures DataFrame is Arctic-safe (no Enums)."""
        df = df.copy()
        for col in df.columns:
            # Convert Enums to values
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.value if isinstance(x, Timeframe) else x
                )
        return df

    def _write(self, symbol: str, df: pd.DataFrame) -> None:
        """Unified write method for all data types."""
        if df.empty:
            return

        df = self._clean_df(df).sort_index()

        # Append if timeseries (bars/events), overwrite if metadata (sec/ca)
        if symbol in ["bars", "events"] and self.lib.has_symbol(symbol):
            self.lib.append(symbol, df)
        else:
            self.lib.write(symbol, df)

    # --- Security Master ---

    @property
    def sec_df(self) -> pd.DataFrame:
        if not self.lib.has_symbol("sec_df"):
            return pd.DataFrame(
                columns=["internal_id", "ticker", "start", "end", "extra"]
            )
        df = self.lib.read("sec_df").data
        # Decode JSON extra field
        if not df.empty and "extra" in df.columns:
            df["extra"] = df["extra"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        return df

    @property
    def ca_df(self) -> pd.DataFrame:
        if not self.lib.has_symbol("ca_df"):
            return pd.DataFrame(
                columns=["internal_id", "ex_date", "type", "value"]
            )
        return self.lib.read("ca_df").data

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
        # For simplicity in demo, read all is fine.
        # Arctic doesn't index sec_df by default like SQL.
        df = self.sec_df
        existing = df[
            (df.ticker == ticker)
            & (df.start <= end_date)
            & (df.end >= start_date)
        ]

        if internal_id is None and not existing.empty:
            return int(existing.iloc[0].internal_id)

        iid = internal_id or self._id_counter
        if internal_id is None:
            self._id_counter += 1

        new_row = pd.DataFrame(
            [asdict(Security(iid, ticker, start_date, end_date, kwargs))]
        )
        new_row["extra"] = new_row["extra"].apply(json.dumps)

        # Metadata is small, overwrite is safer for consistency
        if not df.empty:
            # We must re-encode existing 'extra' before writing back
            df["extra"] = df["extra"].apply(json.dumps)
            full_df = pd.concat([df, new_row], ignore_index=True)
            self._write("sec_df", full_df)
        else:
            self._write("sec_df", new_row)

        return iid

    def get_internal_id(
        self, ticker: str, date: Optional[datetime] = None
    ) -> int:
        return self.register_security(ticker, start=date or datetime.now())

    def get_securities(
        self, tickers: Optional[List[str]] = None
    ) -> List[Security]:
        df = self.sec_df
        records = (df[df.ticker.isin(tickers)] if tickers else df).to_dict(
            "records"
        )
        return [Security(**r) for r in records]

    def get_universe(self, date: datetime) -> List[int]:
        df = self.sec_df
        uids = (
            df[(df.start <= date) & (df.end >= date)]
            .internal_id.unique()
            .tolist()
        )
        return [int(x) for x in uids]

    @property
    def reverse_ism(self) -> Dict[int, str]:
        df = self.sec_df
        return (
            cast(Dict[int, str], df.set_index("internal_id").ticker.to_dict())
            if not df.empty
            else {}
        )

    # --- Data IO ---

    def _as_dict(self, obj: Any) -> Dict[str, Any]:
        d = asdict(obj)
        for k, v in d.items():
            if isinstance(v, Timeframe):
                d[k] = v.value
        return d

    def add_bars(self, bars: List[Bar]) -> None:
        if not bars:
            return
        df = pd.DataFrame([self._as_dict(b) for b in bars]).set_index(
            "timestamp"
        )
        self._write("bars", df)

    def add_events(self, events: List[Event]) -> None:
        df = pd.DataFrame([self._as_dict(e) for e in events])
        if not df.empty:
            df["value"] = df["value"].apply(json.dumps)
        self._write("events", df.set_index("timestamp"))

    def add_ca(self, ca: CorporateAction) -> None:
        new_row = pd.DataFrame([self._as_dict(ca)])
        df = self.ca_df
        if not df.empty:
            df = pd.concat([df, new_row]).drop_duplicates()
        else:
            df = new_row
        self._write("ca_df", df)

    def _query_arctic(
        self,
        iids: List[int],
        start: datetime,
        end: datetime,
        timeframe: Timeframe,
        as_of: Optional[datetime],
    ) -> pd.DataFrame:
        if not self.lib.has_symbol("bars"):
            return pd.DataFrame()

        query = QueryBuilder()
        query = query[
            query.internal_id.isin(iids)
            & (query.timeframe == timeframe.value)
            & (query.timestamp >= start)
            & (query.timestamp <= end)
            & (query.timestamp_knowledge <= (as_of or datetime.now()))
        ]
        return self.lib.read("bars", query_builder=query).data.reset_index()

    def get_bars(self, iids: List[int], cfg: QueryConfig) -> pd.DataFrame:
        # 1. Try exact match
        df = self._query_arctic(
            iids, cfg.start, cfg.end, cfg.timeframe, cfg.as_of
        )

        # 2. Auto-resample fallback
        if (
            df.empty
            and cfg.timeframe != Timeframe.MINUTE
            and cfg.timeframe.is_intraday
        ):
            df = self._query_arctic(
                iids, cfg.start, cfg.end, Timeframe.MINUTE, cfg.as_of
            )
            if not df.empty:
                df = (
                    df.sort_values("timestamp_knowledge")
                    .groupby(["internal_id", "timestamp"])
                    .last()
                    .reset_index()
                )
                df.set_index("timestamp", inplace=True)
                df = (
                    df.groupby("internal_id")
                    .resample(cfg.timeframe.pandas_freq)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                            "timestamp_knowledge": "last",
                        }
                    )
                    .dropna()
                    .reset_index()
                )

        if df.empty:
            return df

        # PIT Deduplication
        if "timestamp" not in df.index.names and "timestamp" in df.columns:
            df = (
                df.sort_values("timestamp_knowledge")
                .groupby(["internal_id", "timestamp"])
                .last()
                .reset_index()
            )

        cols = ["open", "high", "low", "close"]
        df[cols] = df[cols].astype("float64")

        # Corporate Action Adjustments
        ca_df = self.ca_df
        if cfg.adjust and not ca_df.empty:
            # Optimized filtering
            relevant_ca = ca_df[
                ca_df.internal_id.isin(iids) & (ca_df.ex_date <= cfg.end)
            ].sort_values("ex_date", ascending=False)

            for iid in iids:
                adj_factor = 1.0
                for _, r in relevant_ca[
                    relevant_ca.internal_id == iid
                ].iterrows():
                    mask = (df.internal_id == iid) & (df.timestamp < r.ex_date)
                    if r.type == "SPLIT":
                        ratio = 1.0 / r.value
                        df.loc[mask, cols] *= ratio
                        adj_factor *= ratio
                    elif r.type == "DIVIDEND":
                        df.loc[mask, cols] -= r.value * adj_factor

        return df.rename(
            columns={
                c: f"{c}_{cfg.timeframe.value}"
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
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = Timeframe.DAY,
    ) -> None:
        if not self.provider:
            return

        def with_ids(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
            df["internal_id"] = df.apply(
                lambda r: self.get_internal_id(r.ticker, r[date_col]), axis=1
            )
            return df.drop(columns=["ticker"])

        # Bars
        bars = self.provider.fetch_bars(
            tickers, start, end, timeframe=timeframe
        )
        if not bars.empty:
            if "timeframe" in bars.columns:
                tf_series = bars["timeframe"].apply(
                    lambda x: x.value if isinstance(x, Timeframe) else x
                )
            else:
                tf_series = pd.Series([timeframe.value] * len(bars))

            df = with_ids(bars, "timestamp").assign(
                timestamp_knowledge=datetime.now(), timeframe=tf_series
            )
            self._write("bars", df.set_index("timestamp"))

        # Corporate Actions
        ca = self.provider.fetch_corporate_actions(tickers, start, end)
        if not ca.empty:
            if "value" not in ca.columns:
                ca["value"] = ca.apply(
                    lambda r: r.get("ratio") or r.get("amount") or 0.0, axis=1
                )

            # Simple overwrite logic for synced CA
            new_ca = with_ids(ca, "ex_date")[
                ["internal_id", "ex_date", "type", "value"]
            ]
            current_ca = self.ca_df
            combined = pd.concat([current_ca, new_ca]).drop_duplicates()
            self._write("ca_df", combined)

        # Events
        events = self.provider.fetch_events(tickers, start, end)
        if not events.empty:
            df = with_ids(events, "timestamp").assign(
                timestamp_knowledge=datetime.now()
            )
            df["value"] = df["value"].apply(json.dumps)
            self._write("events", df.set_index("timestamp"))
