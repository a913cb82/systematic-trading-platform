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

_ARCTIC_CACHE: Dict[str, Arctic] = {}


class DataPlatform:
    def __init__(
        self,
        *providers: Any,
        db_path: str = "./.arctic_db",
        clear: bool = False,
    ):
        self.providers = providers
        self.arctic = _ARCTIC_CACHE.setdefault(
            db_path, Arctic(f"lmdb://{db_path}")
        )
        if clear and "platform" in self.arctic.list_libraries():
            self.arctic.delete_library("platform")
        self.lib = self.arctic.get_library("platform", create_if_missing=True)
        self.stream_provider = next(
            (p for p in providers if hasattr(p, "subscribe")), None
        )

    def _get_deduplicated_data(
        self, sym: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        symbols_to_dedup = ["bars", "events", "ca_df", "sec_df"]
        if sym in symbols_to_dedup and self.lib.has_symbol(sym):
            ex = self.lib.read(sym).data
            if isinstance(ex.index, pd.DatetimeIndex):
                ex.index = ex.index.tz_localize(None)

            to_concat = []
            if not ex.empty:
                is_range = isinstance(ex.index, pd.RangeIndex)
                to_concat.append(ex.reset_index() if not is_range else ex)
            if not df.empty:
                is_range = isinstance(df.index, pd.RangeIndex)
                to_concat.append(df.reset_index() if not is_range else df)

            if to_concat:
                return pd.concat(to_concat)
        return df

    def _write(self, sym: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(
                    lambda x: x.value if isinstance(x, Timeframe) else x
                )
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c]).dt.tz_localize(None)

        df = self._get_deduplicated_data(sym, df)

        idx_col = next(
            (c for c in ["timestamp", "ex_date"] if c in df.columns), None
        )
        if idx_col:
            # Bitemporal Deduplication: Keep only truly unique versions
            subset = [idx_col, "internal_id"]
            if sym == "bars":
                subset.append("timeframe")
            if "timestamp_knowledge" in df.columns:
                subset.append("timestamp_knowledge")

            # Ensure index column is present for subset check
            df_check = df.reset_index() if idx_col not in df.columns else df
            df = df_check.drop_duplicates(subset=subset, keep="last")
            df = df.set_index(idx_col).sort_index()
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_localize(None)
        else:
            df = df.drop_duplicates().reset_index(drop=True)
        self.lib.write(sym, df)

    def _read(self, sym: str) -> pd.DataFrame:
        if self.lib.has_symbol(sym):
            return self.lib.read(sym).data
        return pd.DataFrame()

    @property
    def sec_df(self) -> pd.DataFrame:
        df = self._read("sec_df")
        if not df.empty:
            df["extra"] = df["extra"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        if df.empty:
            return pd.DataFrame(
                columns=["internal_id", "ticker", "start", "end", "extra"]
            )
        return df

    @property
    def ca_df(self) -> pd.DataFrame:
        df = self._read("ca_df")
        if df.empty:
            return pd.DataFrame(
                columns=["internal_id", "ex_date", "type", "value"]
            )
        return df

    def register_security(
        self,
        ticker: str,
        internal_id: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        **kwargs: Any,
    ) -> int:
        s = (start or datetime(1900, 1, 1)).replace(tzinfo=None)
        e = (end or datetime(2200, 1, 1)).replace(tzinfo=None)
        existing = self.sec_df
        m = existing[
            (existing.ticker == ticker)
            & (existing.start <= e)
            & (existing.end >= s)
        ]
        if not internal_id and not m.empty:
            return int(m.iloc[0].internal_id)
        iid = internal_id or (
            int(existing.internal_id.max() + 1) if not existing.empty else 1000
        )
        sec = Security(
            iid, ticker, s, e, cast(Dict[str, Any], json.dumps(kwargs))
        )
        self._write("sec_df", pd.DataFrame([asdict(sec)]))
        return iid

    def get_internal_id(
        self, ticker: str, date: Optional[datetime] = None
    ) -> int:
        dt = (date or datetime.now()).replace(tzinfo=None)
        return self.register_security(ticker, start=dt)

    def get_securities(
        self, tickers: Optional[List[str]] = None
    ) -> List[Security]:
        df = self.sec_df
        filt = df[df.ticker.isin(tickers)] if tickers else df
        return [Security(**r) for r in filt.to_dict("records")]

    def get_universe(self, date: datetime) -> List[int]:
        df = self.sec_df
        dt = date.replace(tzinfo=None) if date.tzinfo else date
        if df.empty:
            return []
        mask = (pd.to_datetime(df.start).dt.tz_localize(None) <= dt) & (
            pd.to_datetime(df.end).dt.tz_localize(None) >= dt
        )
        return [int(x) for x in df[mask].internal_id.unique()]

    @property
    def reverse_ism(self) -> Dict[int, str]:
        if self.sec_df.empty:
            return {}
        df = self.sec_df.set_index("internal_id")
        return cast(Dict[int, str], df.ticker.to_dict())

    def add_bars(self, bars: List[Bar]) -> None:
        df = pd.DataFrame([asdict(b) for b in bars])
        if not df.empty:
            m = df.internal_id <= 0
            if m.any():
                df.loc[m, "internal_id"] = df[m].apply(
                    lambda r: self.get_internal_id(
                        r.get("_ticker") or "", r.timestamp
                    ),
                    1,
                )
            if "_ticker" in df.columns:
                df = df.drop(columns=["_ticker"])
        self._write("bars", df)

    def add_events(self, evs: List[Event]) -> None:
        df = pd.DataFrame([asdict(e) for e in evs])
        df = df.assign(value=lambda x: x.value.apply(json.dumps))
        self._write("events", df)

    def add_ca(self, ca: CorporateAction) -> None:
        self._write("ca_df", pd.DataFrame([asdict(ca)]))

    def get_bars(self, iids: List[int], cfg: QueryConfig) -> pd.DataFrame:
        def q(tf: Timeframe) -> pd.DataFrame:
            qb = QueryBuilder()
            as_of = (cfg.as_of or datetime.now()).replace(tzinfo=None)
            start = cfg.start.replace(tzinfo=None)
            end = cfg.end.replace(tzinfo=None)
            qb = qb[
                qb.internal_id.isin(iids)
                & (qb.timeframe == tf.value)
                & (qb.timestamp >= start)
                & (qb.timestamp <= end)
            ]
            if not self.lib.has_symbol("bars"):
                return pd.DataFrame()
            df = self.lib.read("bars", query_builder=qb).data.reset_index()
            if not df.empty:
                df["timestamp_knowledge"] = pd.to_datetime(
                    df["timestamp_knowledge"]
                ).dt.tz_localize(None)
                df = df[df.timestamp_knowledge <= as_of]
            return df

        df = q(cfg.timeframe)
        if (
            df.empty
            and cfg.timeframe != Timeframe.MINUTE
            and cfg.timeframe.is_intraday
        ):
            df = q(Timeframe.MINUTE)
            if not df.empty:
                df = (
                    df.sort_values("timestamp_knowledge")
                    .groupby(["internal_id", "timestamp"])
                    .last()
                    .reset_index()
                    .set_index("timestamp")
                )
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
        # Final Point-in-Time deduplication: keep the LATEST version available
        df = (
            df.sort_values("timestamp_knowledge")
            .groupby(["internal_id", "timestamp"])
            .last()
            .reset_index()
        )
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].astype(float)
        if cfg.adjust and not self.ca_df.empty:
            ca = (
                self.ca_df.reset_index()
                if "ex_date" in self.ca_df.index.names
                else self.ca_df
            )
            mask = ca.internal_id.isin(iids) & (ca.ex_date <= cfg.end)
            ca = ca[mask].sort_values("ex_date", ascending=False)
            for iid in iids:
                f = 1.0
                for _, r in ca[ca.internal_id == iid].iterrows():
                    m = (df.internal_id == iid) & (df.timestamp < r.ex_date)
                    if r.type == "SPLIT":
                        rto = 1.0 / r.value
                        df.loc[m, ["open", "high", "low", "close"]] *= rto
                        f *= rto
                    else:
                        target_cols = ["open", "high", "low", "close"]
                        df.loc[m, target_cols] -= r.value * f
        cols = {
            c: f"{c}_{cfg.timeframe.value}"
            for c in ["open", "high", "low", "close", "volume"]
        }
        return df.rename(columns=cols)

    def get_events(
        self,
        iids: List[int],
        types: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Event]:
        qb = QueryBuilder()
        as_of_dt = (as_of or datetime.now()).replace(tzinfo=None)
        qb = qb[qb.internal_id.isin(iids)]
        if not self.lib.has_symbol("events"):
            return []
        df = self.lib.read("events", query_builder=qb).data.reset_index()
        if df.empty:
            return []
        df["timestamp_knowledge"] = pd.to_datetime(
            df["timestamp_knowledge"]
        ).dt.tz_localize(None)
        df = df[df.timestamp_knowledge <= as_of_dt]
        if start:
            df = df[df.timestamp >= start.replace(tzinfo=None)]
        if end:
            df = df[df.timestamp <= end.replace(tzinfo=None)]
        if types:
            df = df[df.event_type.isin(types)]
        # Final PIT deduplication
        df = (
            df.sort_values("timestamp_knowledge")
            .groupby(["internal_id", "timestamp"])
            .last()
            .reset_index()
        )
        return [
            Event(**{**r, "value": json.loads(r["value"])})
            for r in df.to_dict("records")
        ]

    def start_streaming(self, tickers: List[str]) -> None:
        if self.stream_provider:
            self.stream_provider.subscribe(
                tickers, lambda b: self.add_bars([b])
            )
            self.stream_provider.run()

    def sync_data(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = Timeframe.DAY,
    ) -> None:
        def m(df: pd.DataFrame, col: str) -> pd.DataFrame:
            return df.assign(
                internal_id=df.apply(
                    lambda r: self.get_internal_id(r.ticker, r[col]), 1
                )
            ).drop(columns=["ticker"])

        for p in self.providers:
            if hasattr(p, "fetch_bars"):
                b = p.fetch_bars(tickers, start, end, timeframe=timeframe)
                if not b.empty:
                    df = m(b, "timestamp").assign(
                        timestamp_knowledge=datetime.now(),
                        timeframe=timeframe.value,
                    )
                    self._write("bars", df)
            if hasattr(p, "fetch_corporate_actions"):
                ca = p.fetch_corporate_actions(tickers, start, end)
                if not ca.empty:
                    if "value" not in ca.columns:
                        ca["value"] = ca.apply(
                            lambda r: r.get("ratio") or r.get("amount") or 0.0,
                            1,
                        )
                    df = m(ca, "ex_date")[
                        ["internal_id", "ex_date", "type", "value"]
                    ]
                    self._write("ca_df", df)
            if hasattr(p, "fetch_events"):
                ev = p.fetch_events(tickers, start, end)
                if not ev.empty:
                    df = m(ev, "timestamp").assign(
                        timestamp_knowledge=datetime.now()
                    )
                    df = df.assign(value=lambda x: x.value.apply(json.dumps))
                    self._write("events", df)
